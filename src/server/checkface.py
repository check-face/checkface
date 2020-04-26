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

GsInputDim = 512 # updated in worker


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

class LatentProxy:
    '''
    This is an Abstract Base Class for both seeds and guids 
    Represents something that can become a latent, be it a seed or a guid in the database
    '''
    
    def getLatent(self):
        raise NotImplementedError()
    
    def getName(self):
        raise NotImplementedError()
    

class LatentBySeed(LatentProxy):
    def __init__(self, seed: int):
        self.seed = seed
        self.latent = fromSeed(self.seed)

    def getLatent(self):
        return self.latent
    
    def getName(self):
        return f"s{str(self.seed)}"

    def getSeed(self):
        return self.seed
    
    
class LatentByHashString(LatentBySeed):
    def __init__(self, hashstr: str):
        super().__init__(seed=hashToSeed(hashstr))
        self.hashstr = hashstr
        
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
            raise NotImplementedError(f"Latent not implemented for type: {latentType}")
    
    def getLatent(self):
        return self.latent
    
    def getName(self):
        return f"GUID{str(self.guid)}"

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

class EncodeJob:
    def __init__(self, target_image, current_latent, current_ticks, name):
        self.target_image = target_image
        self.current_latent = current_latent
        self.current_ticks = current_ticks
        self.name = name
        self.evt = threading.Event()

    def __str__(self):
        return self.name

    def set_result(self, result_latent):
        self.currentLatent = result_latent
        self.evt.set()

    def get_result(self):
        if self.evt.is_set():
            return True
        else:
            return None

class AlignJob:
    def __init__(self, srcimg, name):
        self.srcimg = srcimg
        self.name = name
        self.evt = threading.Event()

    def __str__(self):
        return self.name

    def set_result(self, aligned_faces_imgs):
        self.aligned_faces_imgs = aligned_faces_imgs
        self.evt.set()

    def wait_for_aligned(self, timeout):
        if self.evt.wait(timeout):
            return self.aligned_faces_imgs
        else:
            return None

default_image_dim = 300

requestTimeSummary = Summary('request_processing_seconds',
                             'Time spent processing request')
imagesGenCounter = Counter('image_generating', 'Number of images generated')
guage_job_queue = Gauge('job_queue', 'Number of jobs in the queue')
guage_align_job_queue = Gauge('align_job_queue', 'Number of jobs in the alignment job queue')
guage_encode_job_queue = Gauge('encode_job_queue', 'Number of jobs in the encode / latent recovery job queue')
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
        latent_type = 'qlatent'
    elif latent_data.shape == (18, 512):
        latent_type = 'dlatent'
    else:
        return flask.Response('Latent must be array of shape (512,) or (18, 512)', status=400)
    guid = uuid.uuid4()
    db.latents.insert_one({'_id':str(guid), 'type': latent_type, 'latent':latent_data.tolist()})
    return str(guid)

job_queue = queue.Queue()
encode_job_queue = queue.Queue()
align_job_queue = queue.Queue()
queues_evt = threading.Event()


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

def handle_generate_image_request(latentProxy: LatentProxy, image_dim):
    name = os.path.join(os.getcwd(), "checkfacedata", "outputImages",
                        f"{latentProxy.getName()}_{image_dim}.jpg")
    os.makedirs(os.path.dirname(name), exist_ok=True)

    if not os.path.isfile(name):
        job = GenerateImageJob(latentProxy.getLatent(), latentProxy.getName())
        job_queue.put(job)
        queues_evt.set()
        guage_job_queue.inc(1)
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
    return handle_generate_image_request(LatentByHashString(hash), 300)

def useHashOrSeedOrGuid(hashstr: str, seedstr: str, guidstr: str):
    if guidstr:
        guid = uuid.UUID(hex = guidstr)
        return LatentByGuid(guid)
    if seedstr:
        try:
            seed = int(seedstr)
            return LatentBySeed(seed)
        except ValueError:
            raise ValueError("Seed must be a base 10 number")

    # fallback on hash if nothing else
    return LatentByHashString(hashstr)

def getRequestLatent(request):
    hashstr = request.args.get('value')
    seedstr = request.args.get('seed')
    guidstr = request.args.get('guid')
    return useHashOrSeedOrGuid(hashstr, seedstr, guidstr)


@app.route('/api/face/', methods=['GET'])
def image_generation():
    with requestTimeSummary.time():
        latentProxy = getRequestLatent(request)
        image_dim = getRequestedImageDim(request)
        return handle_generate_image_request(latentProxy, image_dim)


@app.route('/api/hashdata/', methods=['GET'])
def hashlatentdata():
    latentProxy = getRequestLatent(request)
    data = {"qlatent": latentProxy.getLatent()}
    if isinstance(latentProxy, LatentBySeed):
        data['seed'] = latentProxy.getSeed()

    return jsonify(data)

def generate_gif(fromLatentProxy: LatentProxy, toLatentProxy: LatentProxy, num_frames, fps, image_dim, name):
    latent1 = fromLatentProxy.getLatent()
    latent2 = toLatentProxy.getLatent()

    vals = [(math.sin(i) + 1) * 0.5 for i in np.linspace(0, 2 * math.pi, num_frames, False)]
    latents = [latent1 * i + latent2 * (1 - i) for i in vals]

    jobs = [GenerateImageJob(latent, f"from {fromLatentProxy.getName()} to {toLatentProxy.getName()} f{i}") for i, latent in enumerate(latents)]
    for job in jobs:
        job_queue.put(job)
        queues_evt.set()
        guage_job_queue.inc(1)

    imgs = [job.wait_for_img(30) for job in jobs]

    for img in imgs:
        if not img:
            raise Exception("Generating image failed or timed out")

    resized_imgs = [img.resize((image_dim, image_dim), PIL.Image.ANTIALIAS) for img in imgs]

    resized_imgs[0].save(name, save_all=True, append_images=resized_imgs[1:], duration=1000/fps, loop=0)

@app.route('/api/gif/', methods=['GET'])
def gif_generation():

    fromHash = request.args.get('from_value')
    fromSeedStr = request.args.get('from_seed')
    fromGuidStr = request.args.get('from_guid')
    fromLatentProxy = useHashOrSeedOrGuid(fromHash, fromSeedStr, fromGuidStr)

    toHash = request.args.get('to_value')
    toSeedStr = request.args.get('to_seed')
    toGuidStr = request.args.get('to_guid')
    toLatentProxy = useHashOrSeedOrGuid(toHash, toSeedStr, toGuidStr)

    image_dim = getRequestedImageDim(request)
    num_frames = defaultedRequestInt(request, 'num_frames', 50, 3, 200)
    fps = defaultedRequestInt(request, 'fps', 16, 1, 100)

    name = os.path.join(os.getcwd(), "checkfacedata", "outputGifs",
                        f"from {fromLatentProxy.getName()} to {toLatentProxy.getName()} n{num_frames}f{fps}x{image_dim}.gif")
    os.makedirs(os.path.dirname(name), exist_ok=True)
    

    if not os.path.isfile(name):
        generate_gif(fromLatentProxy, toLatentProxy, num_frames, fps, image_dim, name)
    else:
        print(f"Gif file already exists: {name}")

    return send_file(name, mimetype='image/gif')

def generate_mp4(fromLatentProxy, toLatentProxy, num_frames, fps, kbitrate, image_dim, name):
    latent1 = fromLatentProxy.getLatent()
    latent2 = toLatentProxy.getLatent()

    vals = [(math.sin(i) + 1) * 0.5 for i in np.linspace(0, 2 * math.pi, num_frames, False)]
    latents = [latent1 * i + latent2 * (1 - i) for i in vals]

    jobs = [GenerateImageJob(latent, f"from {fromLatentProxy.getName()} to {toLatentProxy.getName()} f{i}") for i, latent in enumerate(latents)]
    for job in jobs:
        job_queue.put(job)
        queues_evt.set()
        guage_job_queue.inc(1)

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

    fromHash = request.args.get('from_value')
    fromSeedStr = request.args.get('from_seed')
    fromGuidStr = request.args.get('from_guid')
    fromLatentProxy = useHashOrSeedOrGuid(fromHash, fromSeedStr, fromGuidStr)

    toHash = request.args.get('to_value')
    toSeedStr = request.args.get('to_seed')
    toGuidStr = request.args.get('to_guid')
    toLatentProxy = useHashOrSeedOrGuid(toHash, toSeedStr, toGuidStr)

    image_dim = getRequestedImageDim(request)
    num_frames = defaultedRequestInt(request, 'num_frames', 50, 3, 200)
    fps = defaultedRequestInt(request, 'fps', 16, 1, 100)
    kbitrate = defaultedRequestInt(request, 'kbitrate', 2400, 100, 20000)

    name = os.path.join(os.getcwd(), "checkfacedata", "outputMp4s",
                        f"from {fromLatentProxy.getName()} to {toLatentProxy.getName()} n{num_frames}f{fps}x{image_dim}k{kbitrate}.mp4")
    os.makedirs(os.path.dirname(name), exist_ok=True)
    

    if not os.path.isfile(name):
        generate_mp4(fromLatentProxy, toLatentProxy, num_frames, fps, kbitrate, image_dim, name)
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
@app.route('/api/encodeimage/viewimage/', methods=['GET'])
def get_uploadimage():
    imgguid = request.args.get('imgguid')
    record = db.uploadedImages.find_one({'_id': imgguid})
    filename = os.path.join(os.getcwd(), "checkfacedata", "uploadedImages", record["filename"])
    return send_file(filename)

def wait_align_images(srcimg: PIL.Image, name: str):
    job = AlignJob(srcimg, name)
    align_job_queue.put(job)
    queues_evt.set()
    guage_align_job_queue.inc(1)
    imgs = job.wait_for_aligned(30)
    if imgs:
        return imgs
    else:
        raise Exception("Aligning image failed or timed out")

@app.route('/api/encodeimage/alignimage/', methods=['POST'])
def align_uploadedimage():
    imgguid = request.args.get('imgguid')
    srcrecord = db.uploadedImages.find_one({'_id': imgguid})
    if 'alignedguids' in srcrecord:
        print(f"Faces already aligned for {imgguid}")
        return jsonify(srcrecord['alignedguids'])
    else:
        srcfilename = os.path.join(os.getcwd(), "checkfacedata", "uploadedImages", srcrecord["filename"])
        srcimg = PIL.Image.open(srcfilename)
        alignedimgs = wait_align_images(srcimg, f"src_imgguid: {imgguid}")
        alignedguids = []
        try:
            for alignedimg in alignedimgs:
                alignedguid = uuid.uuid4()
                basename = f"{str(alignedguid)}.png"
                filename = os.path.join(os.getcwd(), "checkfacedata", "uploadedImages", basename)
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                alignedimg.save(filename, 'PNG')
                db.uploadedImages.insert_one({'_id': str(alignedguid), 'type': "aligned", 'filename': basename})
                alignedguids.append(str(alignedguid))
                
            db.uploadedImages.update_one(
                {'_id': imgguid},
                {
                    "$set": { "alignedguids": alignedguids }
                })
        except pymongo.errors.PyMongoError:
            return flask.Response('Database error', status=500)
        return jsonify(alignedguids)

@app.route('/api/encodeimage/beginencoding/', methods=['POST'])
def encodemyface():
    imgguid = request.args.get('imgguid')
    record = db.uploadedImages.find_one({'_id': imgguid})
    filename = os.path.join(os.getcwd(), "checkfacedata", "uploadedImages", record["filename"])
    # Begin an encode job for the image previously uploaded to be encoded
    # 
    encodeguid = uuid.uuid4()
    try:
        db.encode_jobs.insert_one({'_id':str(encodeguid) })
    except pymongo.errors.PyMongoError:
        return flask.Response('Database error', status=500)

        # job = GenerateImageJob(latentProxy.getLatent(), latentProxy.getName())
    encode_job_queue.put(job)
    queues_evt.set()
    guage_encode_job_queue.inc(1)

    return str(encodeguid)

    # TODO Implement a second Job Queue
    # TODO Method to make encoding job queue pause temporarily when we hit either 20 
    # requests for a batch, or hit 1s of wait time

    # if not os.path.isfile(name):
    #     job = GenerateImageJob(latentProxy.getLatent(), latentProxy.getName())
    #     job_queue.put(job)
    #     guage_job_queue.inc(1)
    #     img = job.wait_for_img(30)
    #     if img:
    #         resized = img.resize(
    #                 (image_dim, image_dim), PIL.Image.ANTIALIAS)
    #         resized.save(name, 'JPEG')
    #     else:
    #         raise Exception("Generating image failed or timed out")

@app.route('/api/encodeimage/status/', methods=['GET'])
def getmyface():
    encodeguid = request.args.get('encodeguid')
    record = db.encode_jobs.find_one({'_id': encodeguid})
    # filename = os.path.join(os.getcwd(), "checkfacedata", "uploadedImages", record["filename"])
    status = { "finished": False }
    return jsonify(status)

@app.route('/api/queue/', methods=['GET'])
def healthcheck():
    return jsonify({"queue": job_queue.qsize()})


def get_batch(batchsize):
    for i in range(batchsize):
        if not job_queue.empty():
            yield job_queue.get_nowait()
            guage_job_queue.dec(1)
        else:
            break

def create_perceptual_model(Gs):

    from encoder.generator_model import Generator
    from encoder.perceptual_model import PerceptualModel
    # for now it's unclear if larger batch leads to better performance/quality
    batch_size = 1 #@param {type:"slider", min: 1, max: 10, step: 1}

    # Perceptual model params
    image_size= 256 #@param {type:"slider", min: 32, max: 1024, step: 32}

    # Generator params
    randomize_noise = False
    generator = Generator(Gs, batch_size, randomize_noise=randomize_noise)
    perceptual_model = PerceptualModel(image_size, layer=9, batch_size=batch_size)
    perceptual_model.build_perceptual_model(generator.generated_image)
    return generator, perceptual_model

from ffhq_dataset.face_alignment import image_align
# for aligning faces
def load_landmarks_detector():
    import bz2
    from keras.utils import get_file
    from ffhq_dataset.landmarks_detector import LandmarksDetector

    LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'


    def unpack_bz2(src_path):
        dst_path = src_path[:-4]
        if not os.path.exists(dst_path):
            data = bz2.BZ2File(src_path).read()
            with open(dst_path, 'wb') as fp:
                fp.write(data)
        return dst_path

    landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                            LANDMARKS_MODEL_URL, cache_subdir='temp'))
    return LandmarksDetector(landmarks_model_path)

def scaledown(size):
    width, height = size
    ratio = float(width) / height

    # cap so that smallest dimension is at most 512px
    if(width > 512 or height > 512):
        if(width > height):
            height2 = 512
            width2 = ratio * height2
        else:
            width2 = 512
            height2 = width2 / ratio
    else:
        width2 = width
        height2 = height

    # cap at at most 1MP
    maxPixels = 1000000
    pixels = width2 * height2
    if (pixels <= maxPixels):
        return (width2, height2)

    scale = math.sqrt(float(pixels) / maxPixels)
    height3 = int(float(height2) / scale)
    width3 = int(ratio * height2 / scale)
    return (width3, height3)

def align_single_image(landmarks_detector, srcimg: PIL.Image):
    from ffhq_dataset.face_alignment import image_align
    alignedFaces = []
    swidth, sheight = scaledown(srcimg.size) # scale down for landmarks performance
    smafile = os.path.join(os.getcwd(), "checkfacedata", "outputMp4s", "currAlignmentSma.jpg")
    srcimg.resize((swidth, sheight)).save(smafile, "JPEG")
    landmarks = list(landmarks_detector.get_landmarks(smafile))
    for face_landmarks in landmarks:
        
        aligned_image = image_align(srcimg, face_landmarks)
        alignedFaces.append(aligned_image)
    return alignedFaces

def generate_images(Gs):
    if job_queue.empty():
        return False
    generateImageJobs = list(get_batch(int(os.getenv('GENERATOR_BATCH_SIZE', '10'))))

    latents = np.array([job.latent for job in generateImageJobs])

    print(f"Running jobs {[str(job) for job in generateImageJobs]}")

    images = toImages(Gs, [toDLat(Gs, lat) for lat in latents], None)
    for img, job in zip(images, generateImageJobs):
        job.set_result(img)
        imagesGenCounter.inc()

    return True


def align_images(landmarks_detector):
    if align_job_queue.empty(): 
        return False
    alignJob: AlignJob = align_job_queue.get_nowait()
    guage_align_job_queue.dec(1)
    print(f"Running jobs {[str(alignJob)]}")
    aligned_faces = align_single_image(landmarks_detector, alignJob.srcimg)
    alignJob.set_result(aligned_faces)
    print(f"Align job {str(alignJob)} got {len(aligned_faces)} faces")
    # return True

def recover_latents(perceptual_model):
    if encode_job_queue.empty():
        return False
    current_encode_job = encode_job_queue.get_nowait()
    guage_encode_job_queue.dec(1)

    return True

def worker():
    """
    The worker runs on a single thread to execute jobs that require GPU resources
    
    There are 3 basic Jobs that a worker can complete, namely:
     - Alignment
     - Image Generation
     - Image Encoding to latent vectors

    Each of these jobs is put on a queue, we have access to all 3 of the queues
    Priority over system performance with the monolithic architecture

    First we will execte any image batches, as this will return the most things possible
    Execution time can be tracked, as well as the presence of other queue items and
    provide the logic for choosing the cutoff point for latent 
    encoding, and can resume processing a job at a later stage
    Next do any align jobs.
    Followed by any remaining Image Encoding jobs which are by far the most expensive

    """
    dnnlib.tflib.init_tf()
    print("Loading generator model")
    Gs = fetch_model()
    print("Loading landmarks detector")
    landmarks_detector = load_landmarks_detector()
    print("Loading perceptual model")
    perceptual_model = create_perceptual_model(Gs)
    global dlatent_avg
    dlatent_avg = Gs.get_var('dlatent_avg')

    # Setup for the other bits of the program, hacky and vulnerable to race
    # conditions and might have old data
    global GsInputDim
    GsInputDim = Gs.input_shape[1]

    print(f"Warming up generator network with {num_gpus} gpus")
    warmupNetwork = toImages(Gs, np.array([fromSeed(5)]), None)
    print("Generator ready")

    while True:
        hasWork = False

        # Generate any images first
        hasWork = hasWork or generate_images(Gs)

        # Next Align an image job if there are some
        hasWork = hasWork or align_images(landmarks_detector)

        # Finally Encode images and recover the latents
        hasWork = hasWork or recover_latents(perceptual_model)

        # Consider race condition - a job could be added after observing that there is no work but before
        # clearing  queues_evt; we must make sure that the job is handled before waiting on queues_evt
        if hasWork:
            print("Finished batch job")
            queues_evt.set()
        elif queues_evt.is_set():
            print("No work - clearing queues_evt")
            queues_evt.clear()
        else:
            queues_evt.wait() # wait for work trigger to prevent spinning unnecessarily


if __name__ == "__main__":
    t1 = threading.Thread(target=worker, args=[])
    t1.daemon = True # kill thread on program termination (to allow keyboard interrupt)
    t1.start()

    start_http_server(int(os.getenv('METRICS_PORT', '8000')))
    app.run(host="0.0.0.0", port=os.getenv('API_PORT', '8080'))
    print("Closing checkface server")