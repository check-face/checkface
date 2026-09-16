#!/usr/bin/env python3

import flask
import threading
import subprocess
import shutil
import base64
import math
from pathlib import Path
from bson.binary import Binary, UuidRepresentation
from api import load_model
from cpu_threads import thread_status
from original_cache import OriginalCache
from torch_adapter import TorchGenerator
import time
import os
import PIL.Image
import PIL.ImageDraw
import PIL
import sys
import tempfile
import re
import pickle
import numpy as np
import queue
import hashlib
from urllib.parse import parse_qs, urljoin, urlsplit, urlencode
from flask import send_file, request, jsonify, render_template
import flask_cors
from werkzeug.middleware.proxy_fix import ProxyFix
from prometheus_client import start_http_server, Summary, Gauge, Counter
import pymongo
import uuid
import logging
import requests
np.set_printoptions(threshold=np.inf)
DATA_DIR = Path(os.getenv('CHECKFACE_DATA_DIR', str(Path.cwd() / 'checkfacedata'))).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
inference_lock = threading.RLock()
model_ready = threading.Event()
model_adapter = None
original_cache = OriginalCache(DATA_DIR, os.getenv('CHECKFACE_ORIGINAL_CACHE_BYTES', str(10 * 1024**3)))

mongodb_conn_str = os.getenv("MONGODB_CONNECTION_STRING", "mongodb://root:example@db")
client = pymongo.MongoClient(mongodb_conn_str)
db = client.test


def fetch_model():
    return TorchGenerator(load_model(), original_cache=original_cache)

num_gpus = 0  # Portable candidate uses CPU; the legacy parameter remains internal.
synthesis_kwargs = {}

# We need to have access to this dimension to generate qlatents and we don't
# want to have to access the massive Gs object outside of the worker thread,
# thus we update here when we can.

GsInputDim = 512 # updated in worker


def fromSeed(seed):
    return np.random.RandomState(seed).randn(1, GsInputDim)[0]

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


def chooseQorDLat(Gs, latent1, latent2):
    latent1 = np.array(latent1)
    latent2 = np.array(latent2)
    if(latent1.shape[0] == 18 and latent2.shape[0] == 512):
        latent2 = toDLat(Gs, latent2)

    if(latent1.shape[0] == 512 and latent2.shape[0] == 18):
        latent1 = toDLat(Gs, latent1)

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

        app.logger.info(f"Took {diff:.2f} seconds to run {network} on {len(latents)} latents")
        pilImages = [PIL.Image.fromarray(img, 'RGB') for img in images]
        if image_size:
            pilImages = [img.resize(
                (image_size, image_size), PIL.Image.Resampling.LANCZOS)
                for img in pilImages]

        return pilImages

class LatentProxy:
    '''
    This is an Abstract Base Class for both seeds and guids
    Represents something that can become a latent, be it a seed or a guid in the database
    '''

    def getLatent(self, Gs = None):
        raise NotImplementedError()

    def getName(self):
        raise NotImplementedError()

    def getShardPartitions(self):
        raise NotImplementedError()


class LatentBySeed(LatentProxy):
    def __init__(self, seed: int):
        self.seed = seed
        self.latent = fromSeed(self.seed)

    def getLatent(self, Gs = None):
        return self.latent

    def getName(self):
        return f"s{str(self.seed)}"

    def getSeed(self):
        return self.seed

    def getShardPartitions(self):
        return ["s" + str(self.seed % 100), str(self.seed % 10000) ]


class LatentByTextValue(LatentProxy):
    def __init__(self, textValue: str):
        if not textValue:
            textValue = ''
        self.textValue = textValue
        h = hashlib.sha256(textValue.encode('utf-8'))
        self.hashhex = h.hexdigest()

        # https://stackoverflow.com/a/36756272
        # seed is an array of uint32
        seed = np.frombuffer(h.digest(), dtype='uint32')
        self.latent = fromSeed(seed)


    def getLatent(self, Gs = None):
        return self.latent

    def getName(self):
        return f"hash-{str(self.hashhex)}"

    def getHashHex(self):
        return self.hashhex

    def getShardPartitions(self):
        name = self.getName()
        return [  name[:7], name[7:9] ]

class LatentByGuid(LatentProxy):
    def __init__(self, guid: uuid.UUID):
        self.guid = guid
        record = db.latents.find_one({'_id': str(self.guid)})
        if not record:
            # Preserve imported UUID records from Mongo's historical Python codec.
            for representation in (UuidRepresentation.PYTHON_LEGACY, UuidRepresentation.STANDARD):
                record = db.latents.find_one({'_id': Binary.from_uuid(self.guid, uuid_representation=representation)})
                if record:
                    break
        if not record:
            raise KeyError('Cannot find latent for guid')
        latentType = record.get('type', 'qlatent')
        if latentType == 'qlatent':
            self.latent = np.array(record['latent'])
        else:
            self.latent = np.array(record['latent'])
            # raise NotImplementedError(f"Latent not implemented for type: {latentType}")

    def getLatent(self, Gs = None):
        return self.latent

    def getName(self):
        return f"GUID{str(self.guid)}"

    def getShardPartitions(self):
        name = self.getName()
        return [  name[:6], name[6:8] ]

class LatentByLerp(LatentProxy):
    def __init__(self, fromLat:LatentProxy, toLat:LatentProxy, p: float):
        self.fromLat = fromLat
        self.toLat = toLat
        self.p = p

    def getName(self):
        return f"LERP_{self.p:.3f}_{self.fromLat.getName()}-{self.toLat.getName()}_LERP"

    def getShardPartitions(self):
        return [ "LERPS" ]

    def getLatent(self, Gs):
        latent1 = np.array(self.fromLat.getLatent(Gs))
        latent2 = np.array(self.toLat.getLatent(Gs))

        if(latent1.shape[0] == 18 and latent2.shape[0] == 512):
            if not hasattr(self.toLat, 'asDLat'):
                self.toLat.asDLat = toDLat(Gs, latent2)
            latent2 = self.toLat.asDLat

        if(latent1.shape[0] == 512 and latent2.shape[0] == 18):
            if not hasattr(self.fromLat, 'asDLat'):
                self.fromLat.asDLat = toDLat(Gs, latent1)
            latent1 = self.fromLat.asDLat

        return latent1 * (1 - self.p) + latent2 * self.p

class LatentByMultiLerp(LatentProxy):
    def __init__(self, multiLerps):
        self.multiLerps = multiLerps
        names = [latProxy.getName() for [_,latProxy] in self.multiLerps]
        amounts = [f"{p:.3f}" for [p,_] in self.multiLerps]
        middle = "-".join(names) + "_" + "-".join(amounts)
        self.hashhex = hashlib.sha256(middle.encode('utf-8')).hexdigest()

    def getName(self):
        return "MULTILERP_" + self.hashhex + "_MULTILERP"

    def getShardPartitions(self):
        name = self.getName()
        return [ name[:12], name[12:14] ]

    def getLatent(self, Gs):
        latents = [amount * np.array(latProxy.getLatent(Gs)) for [amount,latProxy] in self.multiLerps]

        isAnyDlat = False
        for l in latents:
            if l.shape[0] == 18:
                isAnyDlat = True
                break

        if isAnyDlat:
            for idx, lat in enumerate(latents):
                latProxy = self.multiLerps[idx][1]
                if not hasattr(latProxy, 'asDLat'):
                    latProxy.asDLat = toDLat(Gs, lat)
            latents = [latProxy.asDLat for [_,latProxy] in self.multiLerps]

        return np.sum(latents,0)

class GenerateImageJob:
    def __init__(self, latentproxy, name):
        self.latentproxy = latentproxy
        self.name = name
        self.evt = threading.Event()

    def __str__(self):
        return self.name

    def set_result(self, img):
        self.img = img
        self.evt.set()

    def wait_for_img(self, timeout):
        if self.evt.wait(timeout):
            if getattr(self, 'error', None):
                raise RuntimeError('Generation failed') from self.error
            return self.img
        else:
            return None



default_image_dim = 300

requestTimeSummary = Summary('request_processing_seconds',
                             'Time spent processing request')
imagesGenCounter = Counter('image_generating', 'Number of images generated')
imageEncodedCounter = Counter('image_encoding', 'Number of images encoded')
jobQueue = Gauge('job_queue', 'Number of jobs in the queue')
generatorNetworkTime = Summary('generator_network_seconds', 'Time taken to run \
                                the generator network')
ffmpegTimeSummary = Summary('ffmpeg_processing_seconds',
                             'Time spent running ffmpeg')

app = flask.Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1)
app.config["DEBUG"] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MiB
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 3600 * 24 * 7 # 1 week
flask_cors.CORS(app) # enable CORS so can fetch content

logging.basicConfig(level=logging.INFO)


@app.route('/status/', methods=['GET'])
def status():
    return ''


@app.route('/', methods=['GET'])
def home():
    frontend = Path(os.getenv('CHECKFACE_FRONTEND_DIR', '/app/frontend'))
    if (frontend / 'index.html').is_file():
        return flask.send_from_directory(frontend, 'index.html')
    return 'It works'

def registerLatent(latentArray):
    try:
        latent_data = np.array(latentArray).astype('float32', casting='same_kind')
    except (TypeError, ValueError):
        return (False, 'Latent must be array of floats')
    if not np.isfinite(latent_data).all():
        return (False, 'Latent must contain finite values')
    if latent_data.shape == (512,):
        latent_type = 'qlatent'
    elif latent_data.shape == (18, 512):
        latent_type = 'dlatent'
    else:
        return (False, 'Latent must be array of shape (512,) or (18, 512)')
    guid = uuid.uuid4()
    db.latents.insert_one({'_id':str(guid), 'type': latent_type, 'latent':latent_data.tolist()})
    return (True, str(guid))

@app.route('/api/registerlatent/', methods=['POST'])
def registerLatentApi():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or 'latent' not in payload:
        return flask.Response('JSON latent array is required', status=400)
    latentArray = payload['latent']
    (didSucceed, msg) = registerLatent(latentArray)
    if didSucceed:
        return msg
    else:
        return flask.Response(msg, status=400)

# such a queue
q = queue.Queue(maxsize=int(os.getenv('CHECKFACE_QUEUE_SIZE', '32')))


def enqueue(job):
    q.put(job, timeout=float(os.getenv('CHECKFACE_ENQUEUE_TIMEOUT', '600')))
    jobQueue.inc(1)

def argIsTrue(request, param_name):
    argval = request.args.get(param_name)
    if(argval):
        argval = argval.lower()
    return argval == 'true'

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

def handle_generate_image_request(latentProxy: LatentProxy, image_dim, isWebp):
    partitions = os.path.join(*latentProxy.getShardPartitions())
    imgsDir = os.path.join(DATA_DIR, 'outputImages', partitions)
    os.makedirs(imgsDir, exist_ok=True)

    fileExt = "webp" if isWebp else "jpg"
    fileFormat = "WEBP" if isWebp else "JPEG"
    fileMimetype = "image/webp" if isWebp else "image/jpg"
    name = os.path.join(imgsDir, f"{latentProxy.getName()}_{image_dim}.{fileExt}")
    app.logger.info(f"image file name: {name}")

    if not os.path.isfile(name):
        job = GenerateImageJob(latentProxy, latentProxy.getName())
        enqueue(job)
        img = job.wait_for_img(int(os.getenv('CHECKFACE_JOB_TIMEOUT', '600')))
        if img:
            resized = img.resize(
                    (image_dim, image_dim), PIL.Image.Resampling.LANCZOS)
            save_image_atomic(resized, name, fileFormat)
        else:
            raise Exception("Generating image failed or timed out")


    else:
        app.logger.info(f"Image file already exists: {name}")
    return send_file(name, mimetype=fileMimetype)


@app.route('/api/<string:textValue>', methods=['GET'])
def image_generation_legacy(textValue):
    '''
    string as a type will accept anything without a slash
    path as a type would accept slashes as well

    https://flask.palletsprojects.com/en/1.0.x/quickstart/#variable-rules

    '''
    return handle_generate_image_request(LatentByTextValue(textValue), 300, False)

def useTextOrSeedOrGuid(textValue: str, seedstr: str, guidstr: str):
    if guidstr:
        guid = uuid.UUID(hex = guidstr)
        return LatentByGuid(guid)
    if seedstr:
        try:
            seed = int(seedstr)
            return LatentBySeed(seed)
        except ValueError:
            raise ValueError("Seed must be a base 10 number")

    # fallback on text value if nothing else
    return LatentByTextValue(textValue)

def getMultiLerpLatent(numMulti, request):
    multiLerp = []
    for i in range(numMulti):
        textValue = request.args.get('value' + str(i))
        seedstr = request.args.get('seed' + str(i))
        guidstr = request.args.get('guid' + str(i))
        latentProxy = useTextOrSeedOrGuid(textValue, seedstr, guidstr)
        amountstr = request.args.get('amount' + str(i))
        if amountstr:
            amount = max(-2.0, min(2.0, float(amountstr)))
        else:
            amount = 1.0 / float(numMulti)
        multiLerp.append([amount,latentProxy])
    return LatentByMultiLerp(multiLerp)
    


def getRequestLatent(request):
    numMulti = defaultedRequestInt(request, 'num_multi', 0, 0, 16)
    if numMulti > 0:
        return getMultiLerpLatent(numMulti, request)

    textValue = request.args.get('value')
    seedstr = request.args.get('seed')
    guidstr = request.args.get('guid')
    return useTextOrSeedOrGuid(textValue, seedstr, guidstr)

def getRequestedFormat(request):
    format = request.args.get('format', default='jpg').strip().lower()
    return format


@app.route('/api/face/', methods=['GET'])
def image_generation():
    with requestTimeSummary.time():
        latentProxy = getRequestLatent(request)
        image_dim = getRequestedImageDim(request)
        isWebp = getRequestedFormat(request) == "webp"
        return handle_generate_image_request(latentProxy, image_dim, isWebp)


@app.route('/api/hashdata/', methods=['GET'])
def hashlatentdata():
    latentProxy = getRequestLatent(request)
    with inference_lock:
        latent = latentProxy.getLatent(Gs=model_adapter)
    if latent.shape[0] == 512:
        ltype = "qlatent"
    else:
        ltype = "dlatent"
    data = {ltype: latent.tolist()}
    if isinstance(latentProxy, LatentBySeed):
        data['seed'] = latentProxy.getSeed()
    if isinstance(latentProxy, LatentByTextValue):
        data['hash'] = latentProxy.getHashHex()

    return jsonify(data)

outputMorphsDir = os.path.join(DATA_DIR, 'outputMorphs')
assetsDir = os.path.join(DATA_DIR, 'assets')
os.makedirs(outputMorphsDir, exist_ok=True)

def getParentMorphdir(fromLatentProxy: LatentProxy, toLatentProxy: LatentProxy):

    partitions = fromLatentProxy.getShardPartitions() + toLatentProxy.getShardPartitions()
    partitionsPart = os.path.join(*partitions)
    return os.path.join(outputMorphsDir, partitionsPart,
                    f"from {fromLatentProxy.getName()} to {toLatentProxy.getName()}")

def getFramesMorphdir(parentMorphDir, num_frames, image_dim, isLinear):
    shape = "linear" if isLinear else "trig"
    framesdir = os.path.join(parentMorphDir, "frames",
                    f"{shape} n{num_frames}x{image_dim}")
    return framesdir

def generate_morph_frames(fromLatentProxy: LatentProxy, toLatentProxy: LatentProxy, num_frames, image_dim, framenums, isLinear = False):
    """
    For each specified frame num, checks if the frame exists
    and if necessary generates and saves it. Returns the filenames
    of all required frames, in the same order as framenums.
    Note: does tricks to deduplucate where two frames use same file and
    may have extended frames outside the normal range, so in general you CAN'T
    rely on using glob for ffmpeg for example

    If isLinear, linearly morphs from start to end (inclusive)
    Else, trig morphs from start to end and back (last frame is one frame away from start) (deduplicates if even num_frames)
    """
    parentMorphdir = getParentMorphdir(fromLatentProxy, toLatentProxy)

    framesdir = getFramesMorphdir(parentMorphdir, num_frames, image_dim, isLinear)
    os.makedirs(framesdir, exist_ok=True)

    if (num_frames % 2) == 0 and not isLinear:
        # trig function is mirrored so flip to only first half
        # eg. for num_frames = 10
        # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        # is the same as
        # 0, 1, 2, 3, 4, 5, 4, 3, 2, 1

        framenums = [ i if i < num_frames / 2 or i >= num_frames else num_frames - i for i in framenums ]

    frames = [ (i, os.path.join(framesdir, f"img{i:03d}.jpg")) for i in framenums ]
    filenames = [ fName for i, fName in frames ]
    deduplicateBy = set() # keep all filenames in frames to return, but don't generate same file multiple times
    if isLinear:
        vals = np.linspace(1, 0, num_frames, True) # in reverse to work same as trig
    else:
        vals = [(math.sin(i + math.pi/2) + 1) * 0.5 for i in np.linspace(0, 2 * math.pi, num_frames, False)]

    jobs = []
    if all(os.path.isfile(fName) for fName in filenames):
        if len(filenames) == 1:
            app.logger.info(f"Frame already exists: {filenames[0]}")
        else:
            app.logger.info(f"All required frames already exist in {framesdir}")
        return filenames

    def flush_jobs():
        # Never retain every full-resolution result for a long morph. A batch
        # owns at most four full images; aliases share that image until saved.
        for job, targets in jobs:
            img = job.wait_for_img(int(os.getenv('CHECKFACE_JOB_TIMEOUT', '600')))
            if img is None:
                raise RuntimeError('Generating image failed or timed out')
            try:
                for filename, dimension in targets:
                    resized = img.resize((dimension, dimension), PIL.Image.Resampling.LANCZOS)
                    try:
                        save_image_atomic(resized, filename, 'JPEG')
                    finally:
                        resized.close()
            finally:
                job.img = None
                img.close()
                del img
        jobs.clear()

    for i, fName in frames:
        if os.path.isfile(fName):
            app.logger.info(f'Frame already exists: {fName}')
        elif fName not in deduplicateBy:
            lerpLatentProxy = LatentByLerp(fromLatentProxy, toLatentProxy, 1 - vals[i])
            job = GenerateImageJob(lerpLatentProxy, f'from {fromLatentProxy.getName()} to {toLatentProxy.getName()} n{num_frames}f{i}')
            enqueue(job)
            targets = [(fName, image_dim)]
            deduplicateBy.add(fName)
            if i == 0:
                from_image = os.path.join(parentMorphdir, 'FROM.jpg')
                if not os.path.isfile(from_image):
                    targets.append((from_image, 1024))
            if (isLinear and i == num_frames - 1) or (i == num_frames / 2 and not isLinear):
                to_image = os.path.join(parentMorphdir, 'TO.jpg')
                if not os.path.isfile(to_image):
                    targets.append((to_image, 1024))
            jobs.append((job, targets))
            if len(jobs) == 4:
                flush_jobs()
    flush_jobs()

    return filenames

def generate_link_preview(fromLatentProxy: LatentProxy, toLatentProxy: LatentProxy, preview_width):
    parentMorphdir = getParentMorphdir(fromLatentProxy, toLatentProxy)
    previewsDir = os.path.join(parentMorphdir, "linkPreviews")
    os.makedirs(previewsDir, exist_ok=True)

    name = os.path.join(previewsDir, f"x{preview_width}.jpg")

    if os.path.isfile(name):
        app.logger.info(f"Link preview file already exists: {name}")
        return name

    middleLatentProxy = LatentByLerp(fromLatentProxy, toLatentProxy, 0.5)
    latentProxies = [fromLatentProxy, toLatentProxy, middleLatentProxy]
    jobs = [GenerateImageJob(latentProxy, f"from {fromLatentProxy.getName()} to {toLatentProxy.getName()} preview{i}") for i, latentProxy in enumerate(latentProxies)]
    for job in jobs:
        enqueue(job)

    imgs = [job.wait_for_img(int(os.getenv('CHECKFACE_JOB_TIMEOUT', '600'))) for job in jobs]

    for img in imgs:
        if not img:
            raise Exception("Generating link preview failed or timed out")

    standardHeight = 628
    standardWidth = 1200
    preview_height = int(round(standardHeight/standardWidth * preview_width))

    faceDim = int(round(300 * preview_height / standardHeight))
    sumDim = int(round(512 * preview_height / standardHeight))
    face1 = imgs[0].resize((faceDim, faceDim), PIL.Image.Resampling.LANCZOS)
    face2 = imgs[1].resize((faceDim, faceDim), PIL.Image.Resampling.LANCZOS)
    sumFace = imgs[2].resize((sumDim, sumDim), PIL.Image.Resampling.LANCZOS)

    previewIm = PIL.Image.new("RGB", (preview_width, preview_height), color = "white")

    # add site assets if the exist
    logoAsset = os.path.join(assetsDir, "preview-logo.png")
    sitenameAsset = os.path.join(assetsDir, "preview-sitename.png")
    if os.path.isfile(logoAsset):
        logoImg = PIL.Image.open(logoAsset)
        logoHeight = int(150 * preview_height/standardHeight)
        logoWidth = int(logoImg.size[0] * logoHeight / logoImg.size[1])
        resizedLogo = logoImg.resize((logoWidth, logoHeight), PIL.Image.Resampling.LANCZOS)
        previewIm.paste(resizedLogo, (0, 0))
    if os.path.isfile(sitenameAsset):
        sitenameImg = PIL.Image.open(sitenameAsset)
        sitenameHeight = int(165 * preview_height/standardHeight)
        sitenameWidth = int(sitenameImg.size[0] * sitenameHeight / sitenameImg.size[1])
        resizedSitename = sitenameImg.resize((sitenameWidth, sitenameHeight), PIL.Image.Resampling.LANCZOS)
        sitenameYPos = int(0.5 * (preview_height - faceDim)) + faceDim
        previewIm.paste(resizedSitename, (0, sitenameYPos))


    # add images
    gapsSize = (preview_width - faceDim - faceDim - sumDim) * 0.5 # 2 gaps for plus and equals
    previewIm.paste(face1, (0, int(0.5 * (preview_height - faceDim))))
    previewIm.paste(face2, (int(faceDim + gapsSize), int(0.5 * (preview_height - faceDim))))
    previewIm.paste(sumFace, (int(math.ceil(preview_width - sumDim)), int(0.5 * (preview_height - sumDim))))

    # draw plus sign
    draw = PIL.ImageDraw.Draw(previewIm)
    cwGap1 = faceDim + int(0.5 * gapsSize)
    ch = int(preview_height * 0.5)
    lineWidth = int(round(6 * preview_height / standardHeight))
    symbolSize = 14 * preview_height / standardHeight
    draw.line([cwGap1, ch-symbolSize, cwGap1, ch + symbolSize], width=lineWidth, fill="black")
    draw.line([cwGap1-symbolSize, ch, cwGap1+symbolSize, ch], width=lineWidth, fill="black")

    # draw equals sign
    cwGap2 = preview_width - sumDim - 0.5 * gapsSize
    eqH = int(round(0.6 * symbolSize))
    draw.line([cwGap2-symbolSize, ch - eqH, cwGap2+symbolSize, ch - eqH], width=lineWidth, fill="black")
    draw.line([cwGap2-symbolSize, ch + eqH, cwGap2+symbolSize, ch + eqH], width=lineWidth, fill="black")

    save_image_atomic(previewIm, name, 'JPEG')
    return name

def get_from_latent(request):
    fromTextValue = request.args.get('from_value')
    fromSeedStr = request.args.get('from_seed')
    fromGuidStr = request.args.get('from_guid')
    return useTextOrSeedOrGuid(fromTextValue, fromSeedStr, fromGuidStr)

def get_to_latent(request):
    toTextValue = request.args.get('to_value')
    toSeedStr = request.args.get('to_seed')
    toGuidStr = request.args.get('to_guid')
    return useTextOrSeedOrGuid(toTextValue, toSeedStr, toGuidStr)

def save_image_atomic(image, path, format):
    descriptor, temporary = tempfile.mkstemp(dir=os.path.dirname(path), suffix='.partial')
    os.close(descriptor)
    try:
        image.save(temporary, format)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def ffmpeg_generate_morph_file(filenames, outputFileName, fps=16, kbitrate=2400):
    """Preserve ordered/duplicated legacy frame input and codec options; fail atomically."""
    kind = os.path.splitext(outputFileName)[1]
    descriptor, temporary = tempfile.mkstemp(dir=os.path.dirname(outputFileName), suffix=kind)
    os.close(descriptor)
    with tempfile.NamedTemporaryFile(mode='w+t', delete=False) as concatfile:
        for filename in filenames:
            escaped = filename.replace("'", "'\\''")
            concatfile.write(f"file '{escaped}'\n")
    try:
        command = ['ffmpeg', '-nostdin', '-loglevel', 'error', '-r', str(fps),
                   '-f', 'concat', '-safe', '0', '-i', concatfile.name,
                   '-threads', '2', '-filter_complex_threads', '1']
        if kind == '.gif':
            command += ['-filter_complex', '[0:v] split [a][b];[a] palettegen [p];[b][p] paletteuse']
        elif kind == '.mp4':
            command += ['-b:v', f'{kbitrate}k', '-vcodec', 'libx264']
        elif kind == '.webp':
            command += ['-vcodec', 'libwebp', '-loop', '0']
        else:
            raise ValueError('Unknown morph file format')
        with ffmpegTimeSummary.time():
            subprocess.run(command + ['-y', temporary], check=True,
                           timeout=int(os.getenv('CHECKFACE_FFMPEG_TIMEOUT', '300')))
        if os.path.getsize(temporary) == 0:
            raise RuntimeError('ffmpeg returned an empty output')
        os.replace(temporary, outputFileName)
    finally:
        os.unlink(concatfile.name)
        if os.path.exists(temporary):
            os.unlink(temporary)


@app.route('/api/gif/', methods=['GET'])
def gif_generation():
    fromLatentProxy = get_from_latent(request)
    toLatentProxy = get_to_latent(request)

    image_dim = getRequestedImageDim(request)
    num_frames = defaultedRequestInt(request, 'num_frames', 50, 3, 200)
    fps = defaultedRequestInt(request, 'fps', 16, 1, 100)

    parentMorphdir = getParentMorphdir(fromLatentProxy, toLatentProxy)
    GIFsDir = os.path.join(parentMorphdir, "GIFs")
    name = os.path.join(GIFsDir, f"n{num_frames}f{fps}x{image_dim}.gif")

    if not os.path.isfile(name):
        os.makedirs(GIFsDir, exist_ok=True)
        framenums = np.arange(num_frames)
        filenames = generate_morph_frames(fromLatentProxy, toLatentProxy, num_frames, image_dim, framenums, isLinear=False)
        ffmpeg_generate_morph_file(filenames, name, fps=fps)
    else:
        app.logger.info(f"GIF file already exists: {name}")

    return send_file(name, mimetype='image/gif')

@app.route('/api/mp4/', methods=['GET'])
def mp4_generation():
    fromLatentProxy = get_from_latent(request)
    toLatentProxy = get_to_latent(request)

    image_dim = getRequestedImageDim(request)
    num_frames = defaultedRequestInt(request, 'num_frames', 50, 3, 200)
    fps = defaultedRequestInt(request, 'fps', 16, 1, 100)
    kbitrate = defaultedRequestInt(request, 'kbitrate', 2400, 100, 20000)

    parentMorphdir = getParentMorphdir(fromLatentProxy, toLatentProxy)
    mp4sDir = os.path.join(parentMorphdir, "mp4s")
    name = os.path.join(mp4sDir, f"n{num_frames}f{fps}x{image_dim}k{kbitrate}.mp4")

    if not os.path.isfile(name):
        os.makedirs(mp4sDir, exist_ok=True)
        framenums = np.arange(num_frames)
        filenames = generate_morph_frames(fromLatentProxy, toLatentProxy, num_frames, image_dim, framenums, isLinear=False)
        ffmpeg_generate_morph_file(filenames, name, fps=fps, kbitrate=kbitrate)
    else:
        app.logger.info(f"MP4 file already exists: {name}")


    embed_html = request.args.get('embed_html')
    if(embed_html):
        embed_html = embed_html.lower()
    if embed_html == 'true':
        srcData = "data:video/mp4;base64,"
        with open(name, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            srcData = srcData + encoded_string.decode('utf-8')
        return render_template('mp4.html', title="Rendered mp4", dim=str(image_dim), src=srcData)



    return send_file(name, mimetype='video/mp4', conditional=True)

@app.route('/api/webp/', methods=['GET'])
def webp_generation():
    fromLatentProxy = get_from_latent(request)
    toLatentProxy = get_to_latent(request)

    image_dim = getRequestedImageDim(request)
    num_frames = defaultedRequestInt(request, 'num_frames', 50, 3, 200)
    fps = defaultedRequestInt(request, 'fps', 16, 1, 100)

    parentMorphdir = getParentMorphdir(fromLatentProxy, toLatentProxy)
    webPsDir = os.path.join(parentMorphdir, "webPs")
    name = os.path.join(webPsDir, f"n{num_frames}f{fps}x{image_dim}.webp")

    if not os.path.isfile(name):
        os.makedirs(webPsDir, exist_ok=True)
        framenums = np.arange(num_frames)
        filenames = generate_morph_frames(fromLatentProxy, toLatentProxy, num_frames, image_dim, framenums, isLinear=False)
        ffmpeg_generate_morph_file(filenames, name, fps=fps)
    else:
        app.logger.info(f"WEBP file already exists: {name}")

    return send_file(name, mimetype='image/webp', conditional=True)

@app.route('/api/linkpreview/', methods=['GET'])
def linkpreview_generation():
    fromLatentProxy = get_from_latent(request)
    toLatentProxy = get_to_latent(request)

    preview_width = defaultedRequestInt(request, 'width', 1200, 100, 2400)

    name = generate_link_preview(fromLatentProxy, toLatentProxy, preview_width)
    return send_file(name, mimetype='image/jpg')

@app.route('/api/morphframe/', methods=['GET'])
def morphframe():
    fromLatentProxy = get_from_latent(request)
    toLatentProxy = get_to_latent(request)

    image_dim = getRequestedImageDim(request)
    num_frames = defaultedRequestInt(request, 'num_frames', 50, 3, 200)
    framenum = defaultedRequestInt(request, 'frame_num', 0, 0, num_frames - 1)
    isLinear = argIsTrue(request, 'linear')
    framenums = [framenum]
    filenames = generate_morph_frames(fromLatentProxy, toLatentProxy, num_frames, image_dim, framenums, isLinear)

    return send_file(filenames[0], mimetype='image/jpg')

def encodeRequestKey(imgFile, tryAlign: bool):
    # tryAlign not didAlign, because we want to cache against the request not the result,
    # as we assume the same request will have the same result
    h = hashlib.sha256(imgFile)
    key = h.hexdigest() + f"-tryalign={str(tryAlign)}"

    return key

def getEncodedImagesRecord(requestKey):
    return db.encodedimages.find_one({'_id': requestKey})

def setEncodedImagesRecord(requestKey, guid, didAlign):
    app.logger.info(f"Setting encoded image record for {{'guid': '{str(guid)}', 'did_align': {str(didAlign)}}}")
    db.encodedimages.update_one({'_id': requestKey}, {'$setOnInsert': {'guid': guid, 'did_align': didAlign}}, upsert=True)

@app.route('/api/encodeimage/', methods=['POST'])
def encodeimage():
    file = request.files.get('usrimg')
    if not file:
        return flask.Response('No file uploaded for usrimg', status=400)

    tryAlign = flask.request.form.get('tryalign', 'false')
    tryAlign = tryAlign.lower() == 'true'
    imgFile = file.read()


    # Cache encoding requests by a hash of the image file and value of tryAlign
    requestKey = encodeRequestKey(imgFile, tryAlign)
    existingRecord = getEncodedImagesRecord(requestKey)
    if existingRecord:
        app.logger.info(f"Image encoding for {file.filename} with tryalign={str(tryAlign)} already exists!")
        return flask.jsonify({ 'guid': existingRecord['guid'], 'did_align': existingRecord['did_align'] })

    app.logger.info(f"Encoding image {file.filename} with tryalign={str(tryAlign)}")

    from encoder import encode_image
    with inference_lock:
        # A concurrent identical request may have populated Mongo while we waited.
        existingRecord = getEncodedImagesRecord(requestKey)
        if existingRecord:
            return flask.jsonify({'guid': existingRecord['guid'], 'did_align': existingRecord['did_align']})
        latentArray, didAlign = encode_image(imgFile, tryAlign)
        imageEncodedCounter.inc()
        didSucceed, msg = registerLatent(latentArray)
        if didSucceed:
            guid = msg
            setEncodedImagesRecord(requestKey, guid, bool(didAlign))
            return flask.jsonify({'guid': guid, 'did_align': bool(didAlign)})
        return flask.Response(msg, status=400)

@app.route('/api/encodeimage/', methods=['GET'])
def encodeimageform():
    response = flask.Response(render_template('encode.html'))
    response.headers["Cache-Control"] = "no-cache"
    return response



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


def initialize_runtime():
    global dlatent_avg, GsInputDim, model_adapter
    Gs = fetch_model()
    dlatent_avg = Gs.get_var('dlatent_avg')
    GsInputDim = Gs.input_shape[1]
    model_adapter = Gs
    model_ready.set()
    app.logger.info('Generator ready (CPU)')


def worker():
    Gs = model_adapter
    while True:
        job = q.get()
        jobQueue.dec(1)
        try:
            with inference_lock:
                latent = job.latentproxy.getLatent(Gs)
                generated_before = original_cache.generated
                image = toImages(Gs, [toDLat(Gs, latent)], None)[0]
                imagesGenCounter.inc(original_cache.generated - generated_before)
            job.set_result(image)
        except Exception as error:
            app.logger.exception('Generation job failed')
            job.error = error
            job.set_result(None)
        finally:
            q.task_done()


@app.route('/oembed.json', methods=['GET'])
@app.route('/oembed.json/', methods=['GET'])
def oembed():
    """Port of the existing frontend Server/Index.fs oembed response contract."""
    if request.args.get('format', 'json') != 'json':
        return flask.Response('format not implemented', status=501)
    try:
        dimensions = [int(request.args.get(name, '512')) for name in ('maxwidth', 'maxheight')]
    except ValueError:
        return flask.Response('max dimension not a number', status=400)
    if any(value < 50 or value > 1024 for value in dimensions):
        return flask.Response('max dimension outside of acceptable range', status=400)
    embed_url = request.args.get('url')
    if embed_url is None:
        return flask.Response('oembed url not specified', status=400)
    origin = request.url_root.rstrip('/')
    params = {key: values[-1] for key, values in parse_qs(urlsplit(urljoin(origin + '/', embed_url)).query, keep_blank_values=True).items()}

    def source(prefix):
        # Existing Elmish URL parsing gives values precedence over valid GUIDs/seeds.
        if prefix + 'value' in params:
            value = params[prefix + 'value']
            return 'value', value, value
        if prefix + 'guid' in params:
            try:
                value = str(uuid.UUID(params[prefix + 'guid']))
                return 'guid', value, f'custom ({value[:6]})'
            except ValueError:
                pass
        if prefix + 'seed' in params:
            try:
                value = int(params[prefix + 'seed'])
                if 0 <= value <= 2**32 - 1:
                    return 'seed', value, f'seed {value}'
            except ValueError:
                pass
        return None

    left, right = source('from_'), source('to_')
    dimension = min(dimensions)
    thumb = min(dimension, 128)
    selected = left or ('value', 'hello', 'hello')
    if left and right:
        url = origin + '/api/webp/?' + urlencode({'dim': dimension, 'from_' + left[0]: left[1], 'to_' + right[0]: right[1]})
        thumbnail = origin + '/api/face/?' + urlencode({'dim': dimension, left[0]: left[1], 'format': 'jpg'})
        title = left[2] + ' to ' + right[2]
    else:
        url = origin + '/api/face/?' + urlencode({'dim': dimension, selected[0]: selected[1], 'format': 'webp'})
        thumbnail = origin + '/api/face/?' + urlencode({'dim': thumb, selected[0]: selected[1], 'format': 'jpg'})
        title = selected[2]
    return jsonify({'version': '1.0', 'type': 'photo', 'width': dimension, 'height': dimension,
                    'title': title, 'url': url, 'thumbnail_url': thumbnail,
                    'thumbnail_width': thumb, 'thumbnail_height': thumb,
                    'provider_name': 'facemorph.me', 'provider_url': origin})


@app.route('/<path:filename>', methods=['GET'])
def frontend_file(filename):
    frontend = Path(os.getenv('CHECKFACE_FRONTEND_DIR', '/app/frontend')).resolve()
    target = (frontend / filename).resolve()
    if not target.is_relative_to(frontend) or filename.startswith(('api/', 'healthz', 'status/')):
        flask.abort(404)
    if target.is_file():
        return flask.send_from_directory(frontend, filename)
    if '.' not in Path(filename).name and (frontend / 'index.html').is_file():
        return flask.send_from_directory(frontend, 'index.html')
    flask.abort(404)


# Reserve half the HTTP worker threads for health, queue and cache/metadata checks.
GENERATION_CAPACITY = max(1, min(32, int(os.getenv('CHECKFACE_GENERATION_CAPACITY', '8'))))
request_slots = threading.BoundedSemaphore(GENERATION_CAPACITY)
generation_endpoints = {'image_generation_legacy', 'image_generation', 'gif_generation',
                        'mp4_generation', 'webp_generation', 'linkpreview_generation',
                        'morphframe', 'encodeimage', 'hashlatentdata'}


@app.before_request
def generation_admission():
    if request.endpoint in generation_endpoints:
        if not request_slots.acquire(blocking=False):
            return flask.Response('Generation request capacity reached', status=503, headers={'Retry-After': '1'})
        flask.g.generation_admitted = True


@app.teardown_request
def generation_release(_error):
    if getattr(flask.g, 'generation_admitted', False):
        flask.g.generation_admitted = False
        request_slots.release()


@app.route('/healthz', methods=['GET'])
def readiness():
    return jsonify({'ready': model_ready.is_set(), 'provider': 'cpu', 'queue': q.qsize(), 'generation_capacity': GENERATION_CAPACITY, 'cpu_threads': thread_status(), 'original_cache': {'hits': original_cache.hits, 'generated': original_cache.generated, 'write_failures': original_cache.write_failures}}), (200 if model_ready.is_set() else 503)


@app.errorhandler(queue.Full)
def queue_saturated(_error):
    return flask.Response('Generation queue is full', status=503, headers={'Retry-After': '1'})


@app.errorhandler(ValueError)
def bad_value(error):
    return flask.Response(str(error), status=400)


@app.errorhandler(KeyError)
def missing_value(error):
    return flask.Response(str(error), status=404)


if __name__ == '__main__':
    from waitress import serve
    db.command('ping')
    from encoder import verify_assets
    verify_assets()
    initialize_runtime()
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    start_http_server(int(os.getenv('METRICS_PORT', '8000')))
    serve(app, host='0.0.0.0', port=int(os.getenv('API_PORT', '8080')), threads=GENERATION_CAPACITY * 2,
          connection_limit=max(64, GENERATION_CAPACITY * 4), channel_timeout=30, max_request_body_size=16 * 1024 * 1024)
