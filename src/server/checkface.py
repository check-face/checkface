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
import PIL.ImageDraw
import PIL
import sys
import tempfile
import re
import pickle
import numpy as np
import queue
import hashlib
from flask import send_file, request, jsonify, render_template
import flask_cors
from prometheus_client import start_http_server, Summary, Gauge, Counter
import pymongo
import uuid
import logging
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
                (image_size, image_size), PIL.Image.ANTIALIAS)
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

class LatentByGuid(LatentProxy):
    def __init__(self, guid: uuid.UUID):
        self.guid = guid
        record = db.latents.find_one({'_id': str(self.guid)})
        if not record:
            raise KeyError('Cannot find latent for guid')
        latentType = record['type']
        if latentType == 'qlatent':
            self.latent = np.array(record['latent'])
        else:
            self.latent = np.array(record['latent'])
            # raise NotImplementedError(f"Latent not implemented for type: {latentType}")

    def getLatent(self, Gs = None):
        return self.latent

    def getName(self):
        return f"GUID{str(self.guid)}"

class LatentByLerp(LatentProxy):
    def __init__(self, fromLat:LatentProxy, toLat:LatentProxy, p: float):
        self.fromLat = fromLat
        self.toLat = toLat
        self.p = p

    def getName(self):
        return f"LERP_{p:.3f}_{self.fromLat.getName()}-{self.toLat.getName()}_LERP"

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
ffmpegTimeSummary = Summary('ffmpeg_processing_seconds',
                             'Time spent running ffmpeg')

app = flask.Flask(__name__)
app.config["DEBUG"] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MiB
flask_cors.CORS(app) # enable CORS so can fetch content

logging.basicConfig(level=logging.INFO)


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
        latent_type = 'qlatent'
    elif latent_data.shape == (18, 512):
        latent_type = 'dlatent'
    else:
        return flask.Response('Latent must be array of shape (512,) or (18, 512)', status=400)
    guid = uuid.uuid4()
    db.latents.insert_one({'_id':str(guid), 'type': latent_type, 'latent':latent_data.tolist()})
    return str(guid)

# such a queue
q = queue.Queue()

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
    imgsDir = os.path.join(os.getcwd(), "checkfacedata", "outputImages")
    os.makedirs(imgsDir, exist_ok=True)
    fileExt = "webp" if isWebp else "jpg"
    fileFormat = "WEBP" if isWebp else "JPEG"
    fileMimetype = "image/webp" if isWebp else "image/jpg"
    name = os.path.join(imgsDir, f"{latentProxy.getName()}_{image_dim}.{fileExt}")
    if not os.path.isfile(name):
        job = GenerateImageJob(latentProxy, latentProxy.getName())
        q.put(job)
        jobQueue.inc(1)
        img = job.wait_for_img(30)
        if img:
            resized = img.resize(
                    (image_dim, image_dim), PIL.Image.ANTIALIAS)
            resized.save(name, fileFormat)
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

def getRequestLatent(request):
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
    latent = latentProxy.getLatent(Gs = None) # Only LatentByProxy needs Gs at the moment
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

outputMorphsDir = os.path.join(os.getcwd(), "checkfacedata", "outputMorphs")
assetsDir = os.path.join(os.getcwd(), "checkfacedata", "assets")
os.makedirs(outputMorphsDir, exist_ok=True)

def getParentMorphdir(fromName, toName):
    return os.path.join(outputMorphsDir,
                    f"from {fromName} to {toName}")

def getFramesMorphdir(fromName, toName, num_frames, image_dim, isLinear):
    parentMorphdir = getParentMorphdir(fromName, toName)
    shape = "linear" if isLinear else "trig"
    framesdir = os.path.join(parentMorphdir, "frames",
                    f"{shape} n{num_frames}x{image_dim}")
    return parentMorphdir, framesdir

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
    parentMorphdir, framesdir = getFramesMorphdir(fromLatentProxy.getName(), toLatentProxy.getName(), num_frames, image_dim, isLinear)
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

    for i, fName in frames:
        if os.path.isfile(fName):
            app.logger.info(f"Frame already exists: {fName}")
        elif not (fName in deduplicateBy):
            lerpLatentProxy = LatentByLerp(fromLatentProxy, toLatentProxy, 1 - vals[i])
            job = GenerateImageJob(lerpLatentProxy, f"from {fromLatentProxy.getName()} to {toLatentProxy.getName()} n{num_frames}f{i}")
            q.put(job)
            jobQueue.inc(1)
            jobs.append((job, fName, image_dim))
            deduplicateBy.add(fName)

            # also save full size FROM and TO images for posterity sake
            if i == 0:
                FROM_IMAGE = os.path.join(parentMorphdir, "FROM.jpg")
                if not os.path.isfile(FROM_IMAGE):
                    jobs.append((job, FROM_IMAGE, 1024))

            if (isLinear and i == num_frames - 1) or (i == num_frames / 2 and not isLinear):
                TO_IMAGE = os.path.join(parentMorphdir, "TO.jpg")
                if not os.path.isfile(TO_IMAGE):
                    jobs.append((job, TO_IMAGE, 1024))
    if len(jobs) > 0:
        start = time.time()
        imgs = [(job.wait_for_img(30), fName, dim) for (job, fName, dim) in jobs]
        diff = time.time() - start
        app.logger.info(f"Waited {diff:.2f} seconds for {len(imgs)} morph frames")

        for img, fName, dim in imgs:
            if not img:
                raise Exception("Generating image failed or timed out")
            img.resize((dim, dim), PIL.Image.ANTIALIAS).save(fName, 'JPEG')

    return filenames

def generate_link_preview(fromLatentProxy: LatentProxy, toLatentProxy: LatentProxy, preview_width):
    parentMorphdir = getParentMorphdir(fromLatentProxy.getName(), toLatentProxy.getName())
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
        q.put(job)
        jobQueue.inc(1)

    imgs = [job.wait_for_img(30) for job in jobs]

    for img in imgs:
        if not img:
            raise Exception("Generating link preview failed or timed out")

    standardHeight = 628
    standardWidth = 1200
    preview_height = int(round(standardHeight/standardWidth * preview_width))

    faceDim = int(round(300 * preview_height / standardHeight))
    sumDim = int(round(512 * preview_height / standardHeight))
    face1 = imgs[0].resize((faceDim, faceDim), PIL.Image.ANTIALIAS)
    face2 = imgs[1].resize((faceDim, faceDim), PIL.Image.ANTIALIAS)
    sumFace = imgs[2].resize((sumDim, sumDim), PIL.Image.ANTIALIAS)

    previewIm = PIL.Image.new("RGB", (preview_width, preview_height), color = "white")

    # add site assets if the exist
    logoAsset = os.path.join(assetsDir, "preview-logo.png")
    sitenameAsset = os.path.join(assetsDir, "preview-sitename.png")
    if os.path.isfile(logoAsset):
        logoImg = PIL.Image.open(logoAsset)
        logoHeight = int(150 * preview_height/standardHeight)
        logoWidth = int(logoImg.size[0] * logoHeight / logoImg.size[1])
        resizedLogo = logoImg.resize((logoWidth, logoHeight), PIL.Image.ANTIALIAS)
        previewIm.paste(resizedLogo, (0, 0))
    if os.path.isfile(sitenameAsset):
        sitenameImg = PIL.Image.open(sitenameAsset)
        sitenameHeight = int(165 * preview_height/standardHeight)
        sitenameWidth = int(sitenameImg.size[0] * sitenameHeight / sitenameImg.size[1])
        resizedSitename = sitenameImg.resize((sitenameWidth, sitenameHeight), PIL.Image.ANTIALIAS)
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

    previewIm.save(name, 'JPEG')
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

def ffmpeg_generate_morph_file(filenames, outputFileName, fps=16, kbitrate=2400):
    """
    Take a series of input image filenames and generates a morph file based on the outputFileName extension.
    """
    with tempfile.NamedTemporaryFile(mode='w+t', delete=False) as concatfile:
        concatfile.writelines([ f"file '{filename}'\n" for filename in filenames])
        concatfile.close()
        try:
            with ffmpegTimeSummary.time():
                start = time.time()
                _, kind = os.path.splitext(outputFileName)
                if kind == ".gif":
                    command = f"ffmpeg -r {str(fps)} -f concat -safe 0 -i \"{concatfile.name}\" -filter_complex \"[0:v] split [a][b];[a] palettegen [p];[b][p] paletteuse\" -y \"{outputFileName}\""
                elif kind ==".mp4":
                    command = f"ffmpeg -r {str(fps)} -f concat -safe 0 -i \"{concatfile.name}\" -b {str(kbitrate)}k -vcodec libx264 -y \"{outputFileName}\""
                elif kind == ".webp":
                    command = f"ffmpeg -r {str(fps)} -f concat -safe 0 -i \"{concatfile.name}\" -vcodec libwebp -loop 0 -y \"{outputFileName}\""
                else:
                    raise Exception(f"Unnown kind \"{kind}\" for ffmpeg morph file")

                app.logger.info(command)
                os.system(command)

                diff = time.time() - start
                app.logger.info(f"Took {diff:.2f} seconds running ffmpeg on {len(filenames)} frames for {outputFileName}")
        finally:
            os.unlink(concatfile.name)
            pass

@app.route('/api/gif/', methods=['GET'])
def gif_generation():
    os.makedirs(os.path.join(os.getcwd(), "checkfacedata", "outputGifs"), exist_ok=True)

    fromLatentProxy = get_from_latent(request)
    toLatentProxy = get_to_latent(request)

    image_dim = getRequestedImageDim(request)
    num_frames = defaultedRequestInt(request, 'num_frames', 50, 3, 200)
    fps = defaultedRequestInt(request, 'fps', 16, 1, 100)

    parentMorphdir = getParentMorphdir(fromLatentProxy.getName(), toLatentProxy.getName())
    GIFsDir = os.path.join(parentMorphdir, "GIFs")
    name = os.path.join(GIFsDir, f"n{num_frames}f{fps}x{image_dim}.gif")

    if not os.path.isfile(name):
        os.makedirs(GIFsDir, exist_ok=True)
        framenums = np.arange(num_frames)
        filenames = generate_morph_frames(fromLatentProxy, toLatentProxy, num_frames, image_dim, framenums, isLinear=False)
        ffmpeg_generate_morph_file(filenames, name)
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

    parentMorphdir = getParentMorphdir(fromLatentProxy.getName(), toLatentProxy.getName())
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

    parentMorphdir = getParentMorphdir(fromLatentProxy.getName(), toLatentProxy.getName())
    webPsDir = os.path.join(parentMorphdir, "webPs")
    name = os.path.join(webPsDir, f"n{num_frames}f{fps}x{image_dim}.webp")

    if not os.path.isfile(name):
        os.makedirs(webPsDir, exist_ok=True)
        framenums = np.arange(num_frames)
        filenames = generate_morph_frames(fromLatentProxy, toLatentProxy, num_frames, image_dim, framenums, isLinear=False)
        generate_webp(filenames, name, fps=fps)
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
    framenum = defaultedRequestInt(request, 'frame_num', 0, 0, num_frames)
    isLinear = argIsTrue(request, 'linear')
    framenums = [framenum]
    filenames = generate_morph_frames(fromLatentProxy, toLatentProxy, num_frames, image_dim, framenums, isLinear)

    return send_file(filenames[0], mimetype='image/jpg')

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
    global GsInputDim
    GsInputDim = Gs.input_shape[1]

    app.logger.info(f"Warming up generator network with {num_gpus} gpus")
    warmupNetwork = toImages(Gs, np.array([fromSeed(5)]), None)
    app.logger.info("Generator ready")

    while True:
        generateImageJobs = list(get_batch(int(os.getenv('GENERATOR_BATCH_SIZE', '10'))))

        latents = np.array([job.latentproxy.getLatent(Gs) for job in generateImageJobs])

        app.logger.info(f"Running jobs {[str(job) for job in generateImageJobs]}")

        images = toImages(Gs, [toDLat(Gs, lat) for lat in latents], None)
        for img, job in zip(images, generateImageJobs):
            job.set_result(img)
            imagesGenCounter.inc()

        app.logger.info(f"Finished batch job")



if __name__ == "__main__":
    t1 = threading.Thread(target=worker, args=[])
    t1.daemon = True # kill thread on program termination (to allow keyboard interrupt)
    t1.start()

    start_http_server(int(os.getenv('METRICS_PORT', '8000')))
    app.run(host="0.0.0.0", port=os.getenv('API_PORT', '8080'))
    app.logger.info("Closing checkface server")