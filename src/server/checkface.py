#!/usr/bin/env python3

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
np.set_printoptions(threshold=np.inf)

import flask

sys.path.append('/app/dnnlib')
# dnnlib.tflib.init_tf()

def fetch_model():
    url = 'https://drive.google.com/uc?id=1-O8VHNOpBNHnQyn0yz_pK3PHoc3CboC3'

    with dnnlib.util.open_url(url, cache_dir='cache') as f:
        _G, _D, Gs = pickle.load(f)
        return Gs

synthesis_kwargs = dict(output_transform=dict(
    func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=20)

#We need to have access to this dimension to generate qlatents and we don't want to have to access
#the massive Gs object outside of the worker thread, thus we update here when we can.
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
        lats = Gs.components.mapping.run(np.array([lat]), None)
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
            latents, randomize_noise=False, structure='linear', **synthesis_kwargs)
        network = "synthesis component"
    diff = time.time() - start

    print(f"Took {diff:.2f} seconds to run {network}")
    pilImages = [PIL.Image.fromarray(seed, 'RGB').resize(
        (image_size, image_size), PIL.Image.ANTIALIAS) for seed in images]

    return pilImages


image_dim = 300



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
doneJobSeeds = { "apple", "banana", "orange" }


def handle_generate_image_request(hash):
    os.makedirs("outputImages", exist_ok=True)
    seed = int(hashlib.sha256(hash.encode('utf-8')).hexdigest(), 16) % 10**8
    requested_image = image_dim
    try:
        requested_image = int(request.args.get('dim')) # if key doesn't exist, returns None
        if requested_image is None or requested_image < 10 or requested_image > 1024:
            requested_image = image_dim
    except:
        requested_image = image_dim
    name = os.path.join(os.getcwd(), "outputImages",
        f"s{seed}_{requested_image}.jpg")
    if not os.path.isfile(name):
        q.put((seed, requested_image))
        while not ((seed, requested_image) in doneJobSeeds):
            time.sleep(0.05)
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
    return handle_generate_image_request(hash)

@app.route('/api/face/', methods=['GET'])
def image_generation():
    os.makedirs("outputImages", exist_ok=True)
    hash = request.args.get('value')
    if not hash:
        hash = ''
    return handle_generate_image_request(hash)

@app.route('/api/hashdata/', methods=['GET'])
def hashlatentdata():
    hash = request.args.get('value')
    if not hash:
        hash = ''
    seed = int(hashlib.sha256(hash.encode('utf-8')).hexdigest(), 16) % 10**8
    latent = fromSeed(seed)
    return jsonify({"seed": seed, "qlatent": latent.tolist()})

@app.route('/api/queue/', methods=['GET'])
def healthcheck():
    return jsonify({"queue": q.qsize()})


def worker():
    dnnlib.tflib.init_tf()
    Gs = fetch_model()
    dlatent_avg = Gs.get_var('dlatent_avg')

    #Setup for the other bits of the program, hacky and vulnerable to race conditions and might have old data
    GsInputDim = Gs.input_shape[1]

    while True:
        while q.empty():
            time.sleep(0.05)
            #print('waiting for job')
        else:
            seed, requested_image = q.get()
            name = os.path.join(os.getcwd(), "outputImages", f"s{seed}_{requested_image}.jpg")
            print(f"Running job {seed}_{requested_image}")
            latents = [fromSeed(seed)]
            
            images = toImages(Gs, latents, requested_image)
            images[0].save(name, 'JPEG')
            
            print(f"Finished job {seed}")
            doneJobSeeds.add((seed, requested_image))

t1 = threading.Thread(target=worker, args=[])
t1.start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="443", ssl_context='adhoc')
