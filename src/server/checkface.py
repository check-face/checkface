#!/usr/bin/env python3

from encoder.perceptual_model import PerceptualModel
from encoder.generator_model import Generator
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
np.set_printoptions(threshold=np.inf)

import flask

sys.path.append('/app/dnnlib')
dnnlib.tflib.init_tf()

def fetch_model():
    url = 'https://drive.google.com/uc?id=1-O8VHNOpBNHnQyn0yz_pK3PHoc3CboC3'

    with dnnlib.util.open_url(url, cache_dir='cache') as f:
        _G, _D, Gs = pickle.load(f)
        return Gs

synthesis_kwargs = dict(output_transform=dict(
    func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=20)

Gs = fetch_model()


def fromSeed(seed, dim=0, dimOffset=0):
    qlat = np.random.RandomState(seed).randn(1, Gs.input_shape[1])[0]
    if dimOffset != 0:
        qlat[dim] += dimOffset
    return qlat


dlatent_avg = Gs.get_var('dlatent_avg')


def truncTrick(dlatents, psi=0.7, cutoff=8):
    #   return (toDLat(lat) - dlatent_avg) * psi + avg
    layer_idx = np.arange(18)[np.newaxis, :, np.newaxis]
    ones = np.ones(layer_idx.shape, dtype=np.float32)
    coefs = np.where(layer_idx < cutoff, psi * ones, ones)
    dlatents = (dlatents - dlatent_avg) * coefs + dlatent_avg
    return dlatents

def toDLat(lat, useTruncTrick=True):
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


def toImages(latents, image_size):
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


image_dim = 1024

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/', methods=['GET'])
def home():
    return 'It works'


@app.route('/api/<path:hash>', methods=['GET'])
def image_generation(hash):
    os.makedirs("outputImages", exist_ok=True)
    seed = int(hashlib.sha256(hashSeed.encode('utf-8')).hexdigest(), 16) % 10**8
    
    latents = [fromSeed(seed)]
    images = toImages(latents, image_dim)

    name = f"s{seed}.jpg"
    images[0].save(os.path.join("outputImages", name), 'JPEG')
    print("Saved images")

app.run(host="0.0.0.0", port="80")
    