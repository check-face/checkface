#!/usr/bin/env python3

import flask
import threading
import subprocess
import shutil
import base64
import math
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
from flask import send_file, request, jsonify, render_template
from prometheus_client import start_http_server, Summary, Gauge, Counter
import pymongo
import uuid
np.set_printoptions(threshold=np.inf)
client = pymongo.MongoClient("mongodb://root:example@db")
db = client.test


sys.path.append('/app/dnnlib')
# dnnlib.tflib.init_tf()


def fetch_model():
    network_pkl = 'gdrive:networks/stylegan2-ffhq-config-f.pkl'
    import pretrained_networks
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    return Gs
num_gpus = int(os.getenv('NUM_GPUS', '1'))
synthesis_kwargs = dict(output_transform=dict(
    func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=20, num_gpus=num_gpus)

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
    with generatorNetworkTime.time():
        start = time.time()
        if(isinstance(latents, list)):
            isDlat = False
            for lat in latents:
                if lat.shape[0] == 18:
                    isDlat = True
                    break
            if isDlat:
                latents = [toDLat(Gs, lat) for lat in latents]

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
    def __init__(self, latent, name):
        self.latent = latent
        self.name = name
        self.evt = threading.Event()

    def __str__(self):
        return self.name

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
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MiB


@app.route('/status/', methods=['GET'])
def status():
    return ''


@app.route('/', methods=['GET'])
def home():
    return 'It works'

@app.route('/api/registerlatent/', methods=['POST'])
def registerLatent():
    data = request.json
    try:
        latent_data = np.array(data['latent']).astype('float32', casting='same_kind')
    except TypeError:
        return flask.Response('Latent must be array of floats', status=400)
    if latent_data.shape == (512,):
        latent_type = 'dlatent'
    elif latent_data.shape == (18, 512):
        latent_type = 'qlatent'
    else:
        return flask.Response('Latent must be array of shape (512,) or (18, 512)', status=400)
    guid = uuid.uuid4()
    db.latents.insert_one({'_id':str(guid), 'type': latent_type, 'latent':latent_data.tolist()})
    return str(guid)

# such a queue
q = queue.Queue()


def defaultedRequestInt(request, param_name, default_val, min_val, max_val):
    val = default_val
    try:
        # if key doesn't exist, returns None
        val = int(request.args.get(param_name))
        if (val is None or
                val < min_val or
                val > max_val):
            val = default_val
    except:
        val = default_val
    return val

def getRequestedImageDim(request):
    return defaultedRequestInt(request, 'dim', default_image_dim, 10, 1024)

def handle_generate_image_request(seed, image_dim):
    os.makedirs("outputImages", exist_ok=True)

    name = os.path.join(os.getcwd(), "outputImages",
                        f"s{seed}_{image_dim}.jpg")
    if not os.path.isfile(name):
        job = GenerateImageJob(fromSeed(seed), f"s{seed}_{image_dim}")
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
        print(f"Image file already exists: {name}")
    return send_file(name, mimetype='image/jpg')


@app.route('/api/<string:hash>', methods=['GET'])
def image_generation_legacy(hash):
    '''
    string as a type will accept anything without a slash
    path as a type would accept slashes as well

    https://flask.palletsprojects.com/en/1.0.x/quickstart/#variable-rules

    '''
    return handle_generate_image_request(hashToSeed(hash))

def useHashOrSeed(hash, seedstr):
    if seedstr:
        seed = int(seedstr)
    else:
        seed = hashToSeed(hash)
    return seed

def getRequestSeed(request):
    hash = request.args.get('value')
    seedstr = request.args.get('seed')
    return useHashOrSeed(hash, seedstr)


@app.route('/api/face/', methods=['GET'])
def image_generation():
    with requestTimeSummary.time():
        seed = getRequestSeed(request)
        image_dim = getRequestedImageDim(request)
        return handle_generate_image_request(seed, image_dim)


@app.route('/api/hashdata/', methods=['GET'])
def hashlatentdata():
    seed = getRequestSeed(request)
    latent = fromSeed(seed)
    return jsonify({"seed": seed, "qlatent": latent.tolist()})

def generate_gif(from_seed, to_seed, num_frames, fps, image_dim, name):
    latent1 = fromSeed(from_seed)
    latent2 = fromSeed(to_seed)

    vals = [(math.sin(i) + 1) * 0.5 for i in np.linspace(0, 2 * math.pi, num_frames, False)]
    latents = [latent1 * i + latent2 * (1 - i) for i in vals]

    jobs = [GenerateImageJob(latent, f"from s{from_seed} to s{to_seed} f{i}") for i, latent in enumerate(latents)]
    for job in jobs:
        q.put(job)
        jobQueue.inc(1)

    imgs = [job.wait_for_img(30) for job in jobs]

    for img in imgs:
        if not img:
            raise Exception("Generating image failed or timed out")

    resized_imgs = [img.resize((image_dim, image_dim), PIL.Image.ANTIALIAS) for img in imgs]

    resized_imgs[0].save(name, save_all=True, append_images=resized_imgs[1:], duration=1000/fps, loop=0)

@app.route('/api/gif/', methods=['GET'])
def gif_generation():
    os.makedirs("outputGifs", exist_ok=True)

    fromHash = request.args.get('from_value')
    fromSeedStr = request.args.get('from_seed')
    from_seed = useHashOrSeed(fromHash, fromSeedStr)

    toHash = request.args.get('to_value')
    toSeedStr = request.args.get('to_seed')
    to_seed = useHashOrSeed(toHash, toSeedStr)

    image_dim = getRequestedImageDim(request)
    num_frames = defaultedRequestInt(request, 'num_frames', 50, 3, 200)
    fps = defaultedRequestInt(request, 'fps', 16, 1, 100)

    name = os.path.join(os.getcwd(), "outputGifs",
                        f"from s{from_seed} to s{to_seed} n{num_frames}f{fps}x{image_dim}.gif")
    if not os.path.isfile(name):
        generate_gif(from_seed, to_seed, num_frames, fps, image_dim, name)
    else:
        print(f"Gif file already exists: {name}")

    return send_file(name, mimetype='image/gif')

def generate_mp4(from_seed, to_seed, num_frames, fps, kbitrate, image_dim, name):
    latent1 = fromSeed(from_seed)
    latent2 = fromSeed(to_seed)

    vals = [(math.sin(i) + 1) * 0.5 for i in np.linspace(0, 2 * math.pi, num_frames, False)]
    latents = [latent1 * i + latent2 * (1 - i) for i in vals]

    jobs = [GenerateImageJob(latent, f"from s{from_seed} to s{to_seed} f{i}") for i, latent in enumerate(latents)]
    for job in jobs:
        q.put(job)
        jobQueue.inc(1)

    imgs = [job.wait_for_img(30) for job in jobs]

    for img in imgs:
        if not img:
            raise Exception("Generating image failed or timed out")

    framesdir = name + " - frames"
    os.makedirs(framesdir, exist_ok=True)
    for i, img in enumerate(imgs):
        img.resize((image_dim, image_dim), PIL.Image.ANTIALIAS).save(os.path.join(framesdir, f"img{i:03d}.jpg"), 'JPEG')

    print(f"ffmpeg -r {str(fps)} -i \"{framesdir}/img%03d.jpg\" -b {str(kbitrate)}k -vcodec libx264 -y \"{name}\"")
    os.system(f"ffmpeg -r {str(fps)} -i \"{framesdir}/img%03d.jpg\" -b {str(kbitrate)}k -vcodec libx264 -y \"{name}\"")


@app.route('/api/mp4/', methods=['GET'])
def mp4_generation():
    os.makedirs("outputMp4s", exist_ok=True)

    fromHash = request.args.get('from_value')
    fromSeedStr = request.args.get('from_seed')
    from_seed = useHashOrSeed(fromHash, fromSeedStr)

    toHash = request.args.get('to_value')
    toSeedStr = request.args.get('to_seed')
    to_seed = useHashOrSeed(toHash, toSeedStr)

    image_dim = getRequestedImageDim(request)
    num_frames = defaultedRequestInt(request, 'num_frames', 50, 3, 200)
    fps = defaultedRequestInt(request, 'fps', 16, 1, 100)
    kbitrate = defaultedRequestInt(request, 'kbitrate', 2400, 100, 20000)

    name = os.path.join(os.getcwd(), "outputMp4s",
                        f"from s{from_seed} to s{to_seed} n{num_frames}f{fps}x{image_dim}k{kbitrate}.mp4")

    if not os.path.isfile(name):
        generate_mp4(from_seed, to_seed, num_frames, fps, kbitrate, image_dim, name)
    else:
        print(f"MP4 file already exists: {name}")


    embed_html = request.args.get('embed_html')
    if(embed_html):
        embed_html = embed_html.lower()
    if embed_html == 'true':
        srcData = "data:video/mp4;base64,"
        with open(name, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            srcData = srcData + encoded_string.decode('utf-8')
        return render_template('mp4.html', title="Rendered mp4", dim=str(image_dim), src=srcData)



    return send_file(name, mimetype='video/mp4')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def getextension(filename):
    return filename.rsplit('.', 1)[1].lower()
def allowed_file(filename):
    return '.' in filename and \
           getextension(filename) in ALLOWED_EXTENSIONS

@app.route('/api/uploadimage/', methods=['POST'])
def uploadimage():
    file = request.files['usrimg']
    if not file:
        return flask.Response('No file uploaded for usrimg', status=400)
    elif not allowed_file(file.filename):
        return flask.Response(f'File extension must be in {ALLOWED_EXTENSIONS}', status=400)
    else:
        guid = uuid.uuid4()
        basename = f"{str(guid)}.{getextension(file.filename)}"
        filename = os.path.join(os.getcwd(), "checkfacedata", "uploadedImages", basename)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        file.save(filename)
        try:
            db.uploadedImages.insert_one({'_id':str(guid), 'type': "rawupload", 'filename': basename})
        except pymongo.errors.PyMongoError:
            return flask.Response('Database error', status=500)
        return str(guid)

@app.route('/api/uploadimage/', methods=['GET'])
def getmyface():
    imgguid = request.args.get('imgguid')
    record = db.uploadedImages.find_one({'_id': imgguid})
    filename = os.path.join(os.getcwd(), "checkfacedata", "uploadedImages", record["filename"])
    return send_file(filename)


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
    global dlatent_avg
    dlatent_avg = Gs.get_var('dlatent_avg')

    # Setup for the other bits of the program, hacky and vulnerable to race
    # conditions and might have old data
    GsInputDim = Gs.input_shape[1]

    print(f"Warming up generator network with {num_gpus} gpus")
    warmupNetwork = toImages(Gs, np.array([fromSeed(5)]), None)
    print("Generator ready")

    while True:
        generateImageJobs = list(get_batch(int(os.getenv('GENERATOR_BATCH_SIZE', '10'))))

        latents = np.array([job.latent for job in generateImageJobs])

        print(f"Running jobs {[str(job) for job in generateImageJobs]}")

        images = toImages(Gs, [toDLat(Gs, lat) for lat in latents], None)
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