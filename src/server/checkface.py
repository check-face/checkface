#!/usr/bin/env python3

import flask
from encoder.perceptual_model import PerceptualModel
from encoder.generator_model import Generator
import threading
import subprocess
import shutil
import base64
import math
import config
import dnnlib.tflib as tflib
import dnnlib
import time
import os
import PIL.Image
import PIL
import sys
import re
import pickle
import numpy as np
import queue
import hashlib
from flask import send_file, request, jsonify
from prometheus_client import start_http_server, Summary, Gauge, Counter
np.set_printoptions(threshold=np.inf)


sys.path.append('/app/dnnlib')
# dnnlib.tflib.init_tf()


def fetch_model():
    url = 'https://drive.google.com/uc?id=1-O8VHNOpBNHnQyn0yz_pK3PHoc3CboC3'

    with dnnlib.util.open_url(url, cache_dir='cache') as f:
        _G, _D, Gs = pickle.load(f)
        return Gs

synthesis_kwargs = dict(output_transform=dict(
    func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=20)

# We need to have access to this dimension to generate qlatents and we don't
# want to have to access the massive Gs object outside of the worker thread,
# thus we update here when we can.

GsInputDim = 512


def fromSeed(seed, dim=0, dimOffset=0):
    qlat = np.random.RandomState(seed).randn(1, GsInputDim)[0]
    if dimOffset != 0:
        qlat[dim] += dimOffset
    return qlat


dlatent_avg = ""


def truncTrick(dlatents, psi=0.7, cutoff=8):
    #   return (toDLat(lat) - dlatent_avg) * psi + avg
    layer_idx = np.arange(18)[np.newaxis, :, np.newaxis]
    ones = np.ones(layer_idx.shape, dtype=np.float32)
    coefs = np.where(layer_idx < cutoff, psi * ones, ones)
    dlatents = (dlatents - dlatent_avg) * coefs + dlatent_avg
    return dlatents


def toDLat(Gs, lat, useTruncTrick=True):
    lat = np.array(lat)
    if lat.shape[0] == 512:
        lats = Gs.components.mappinjobQueuerun(np.array([lat]), None)
        if useTruncTrick:
            lat = truncTrick(lats)[0]
        else:
            lat = lats[0]
    return lat


def chooseQorDLat(latent1, latent2):
    latent1 = np.array(latent1)
    latent2 = np.array(latent2)
    if(latent1.shape[0] == 18 and latent2.shape[0] == 512):
        latent2 = toDLat(latent2)

    if(latent1.shape[0] == 512 and latent2.shape[0] == 18):
        latent1 = toDLat(latent1)

    return latent1, latent2


def toImages(Gs, latents, image_size):
    with generatorNetworkTime.time():
        start = time.time()
        if(isinstance(latents, list)):
            isDlat = False
            for lat in latents:
                if lat.shape[0] == 18:
                    isDlat = True
                    break
            if isDlat:
                latents = [toDLat(lat) for lat in latents]

        latents = np.array(latents)
        if latents.shape[1] == 512:
            images = Gs.run(latents, None, **synthesis_kwargs)
            network = "generator network"
        else:
            images = Gs.components.synthesis.run(
                latents, randomize_noise=False, structure='linear',
                **synthesis_kwargs)
            network = "synthesis component"
        diff = time.time() - start

        print(f"Took {diff:.2f} seconds to run {network}")
        pilImages = [PIL.Image.fromarray(img, 'RGB') for img in images]
        if image_size:
            pilImages = [img.resize(
                (image_size, image_size), PIL.Image.ANTIALIAS)
                for img in pilImages]

        return pilImages

def hashToSeed(hash):
    if not hash:
        hash = ''
    return int(hashlib.sha256(hash.encode('utf-8')).hexdigest(), 16) % 10**8

class GenerateImageJob:
    def __init__(self, seed, image_dim):
        self.seed = seed
        self.image_dim = image_dim
        self.evt = threading.Event()

    def __str__(self):
        return f"s{self.seed}_{self.image_dim}"

    def set_result(self, img):
        self.img = img
        self.evt.set()

    def wait_for_img(self, timeout):
        if self.evt.wait(timeout):
            return self.img
        else:
            return None



default_image_dim = 300

requestTimeSummary = Summary('request_processing_seconds',
                             'Time spent processing request')
imagesGenCounter = Counter('image_generating', 'Number of images generated')
jobQueue = Gauge('job_queue', 'Number of jobs in the queue')
generatorNetworkTime = Summary('generator_network_seconds', 'Time taken to run \
                                the generator network')

app = flask.Flask(__name__)
app.config["DEBUG"] = False


@app.route('/status/', methods=['GET'])
def status():
    return ''


@app.route('/', methods=['GET'])
def home():
    return 'It works'


# such a queue
q = queue.Queue()


def handle_generate_image_request(seed):
    os.makedirs("outputImages", exist_ok=True)
    image_dim = default_image_dim
    try:
        # if key doesn't exist, returns None
        image_dim = int(request.args.get('dim'))
        if (image_dim is None or
                image_dim < 10 or
                image_dim > 1024):
            image_dim = default_image_dim
    except:
        image_dim = default_image_dim
    name = os.path.join(os.getcwd(), "outputImages",
                        f"s{seed}_{image_dim}.jpg")
    if not os.path.isfile(name):
        job = GenerateImageJob(seed, image_dim)
        q.put(job)
        jobQueue.inc(1)
        img = job.wait_for_img(30)
        if img:
            resized = img.resize(
                    (image_dim, image_dim), PIL.Image.ANTIALIAS)
            resized.save(name, 'JPEG')
        else:
            raise Exception("Generating image failed or timed out")

        
    else:
        print(f"Image file {name} already exists")
    return send_file(name, mimetype='image/jpg')


@app.route('/api/<string:hash>', methods=['GET'])
def image_generation_legacy(hash):
    '''
    string as a type will accept anything without a slash
    path as a type would accept slashes as well

    https://flask.palletsprojects.com/en/1.0.x/quickstart/#variable-rules

    '''
    return handle_generate_image_request(hashToSeed(hash))

def getRequestSeed(request):
    hash = request.args.get('value')
    seedstr = request.args.get('seed')
    if seedstr:
        seed = int(seedstr)
    else:
        seed = hashToSeed(hash)
    return seed


@app.route('/api/face/', methods=['GET'])
def image_generation():
    with requestTimeSummary.time():
        seed = getRequestSeed(request)
        return handle_generate_image_request(seed)


@app.route('/api/hashdata/', methods=['GET'])
def hashlatentdata():
    seed = getRequestSeed(request)
    latent = fromSeed(seed)
    return jsonify({"seed": seed, "qlatent": latent.tolist()})


@app.route('/api/queue/', methods=['GET'])
def healthcheck():
    return jsonify({"queue": q.qsize()})


def get_batch(batchsize):
    yield q.get(True) # will block until it gets a job
    jobQueue.dec(1)
    for i in range(batchsize-1):
        if not q.empty():
            yield q.get_nowait()
            jobQueue.dec(1)


def worker():
    dnnlib.tflib.init_tf()
    Gs = fetch_model()
    dlatent_avg = Gs.get_var('dlatent_avg')

    # Setup for the other bits of the program, hacky and vulnerable to race
    # conditions and might have old data
    GsInputDim = Gs.input_shape[1]

    print("Warming up generator network")
    warmupNetwork = toImages(Gs, np.array([fromSeed(5)]), None)
    print("Generator ready")

    while True:
        generateImageJobs = list(get_batch(int(os.getenv('GENERATOR_BATCH_SIZE', '20'))))
        
        seeds = [job.seed for job in generateImageJobs]

        print(f"Running jobs {[str(job) for job in generateImageJobs]}")
        latents = np.array([fromSeed(seed) for seed in seeds])

        images = toImages(Gs, latents, None)
        for img, job in zip(images, generateImageJobs):
            job.set_result(img)
            imagesGenCounter.inc()

        print(f"Finished batch job")



if __name__ == "__main__":
    t1 = threading.Thread(target=worker, args=[])
    t1.daemon = True # kill thread on program termination (to allow keyboard interrupt)
    t1.start()

    start_http_server(int(os.getenv('METRICS_PORT', '8000')))
    app.run(host="0.0.0.0", port=os.getenv('API_PORT', '8080'))
    print("Closing checkface server")
