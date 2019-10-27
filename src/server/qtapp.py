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

output_path = r"C:\Users\Oliver\Source\Repos\stylegan\results\outgifs"
def fetch_model():
    snap = r"C:\Users\Oliver\Source\Repos\stylegan\results\00004-sgan-flower-1gpu\network-snapshot-008761.pkl"

    with open(snap, 'rb') as f:
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
    layer_idx = np.arange(synthesis_layers)[np.newaxis, :, np.newaxis]
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
    if(latent1.shape[0] == synthesis_layers and latent2.shape[0] == 512):
        latent2 = toDLat(latent2)

    if(latent1.shape[0] == 512 and latent2.shape[0] == synthesis_layers):
        latent1 = toDLat(latent1)

    return latent1, latent2


def toImages(Gs, latents, image_size):

    start = time.time()
    if(isinstance(latents, list)):
        isDlat = False
        for lat in latents:
            if lat.shape[0] == synthesis_layers:
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

# import sys
# from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QSlider
# from PyQt5.QtGui import QPixmap

# app = QApplication([])
# win = QMainWindow()
# label = QLabel(win)
# pixmap = QPixmap('outputImages\s13_300.jpg')
# label.setPixmap(pixmap)
# win.setCentralWidget(label)
# win.show()
# app.exit(app.exec_())

import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QCheckBox, QGridLayout, QGroupBox, QLabel,
                             QMenu, QPushButton, QRadioButton, QVBoxLayout, QWidget, QSlider)
from PyQt5.QtGui import QPixmap, QImage
from PIL.ImageQt import ImageQt

dnnlib.tflib.init_tf()
Gs = fetch_model()
synthesis_layers = Gs.components.synthesis.input_shape[1]
dlatent_avg = Gs.get_var('dlatent_avg')

class Window(QWidget):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        grid = QGridLayout()
        
        self.imlabel = QLabel()
        self.lat1seed = 13
        self.lat2seed = 13
        self.lerpval = 0
        grid.addWidget(self.imlabel, 0, 0)

        slider1 = self.createSlider()
        slider1.valueChanged.connect(self.slider1changed)
        grid.addWidget(slider1, 1, 0)

        slider2 = self.createSlider()
        slider2.valueChanged.connect(self.slider2changed)
        grid.addWidget(slider2, 2, 0)

        slider3 = self.createSlider()
        slider3.valueChanged.connect(self.slider3changed)
        grid.addWidget(slider3, 3, 0)
        
        button = QPushButton('Create gif')
        button.clicked.connect(self.createGif)
        grid.addWidget(button, 4, 0)
        
        self.setLayout(grid)

        self.setWindowTitle("Checkface Sliders")
        #self.resize(400, 300)

        self.renderImage()

    def createSlider(self):
        slider = QSlider(Qt.Horizontal)
        slider.setFocusPolicy(Qt.StrongFocus)
        slider.setTickPosition(QSlider.TicksBothSides)
        slider.setTickInterval(10)
        slider.setSingleStep(1)
        return slider

    def slider1changed(self, val):
        self.lat1seed = val
        self.lerpval = 0
        self.renderImage()

    def slider2changed(self, val):
        self.lat2seed = val
        self.lerpval = 1
        self.renderImage()

    def slider3changed(self, val):
        self.lerpval = val / 100
        self.renderImage()

    def renderImage(self):
        print(f"s{self.lat1seed} to s{self.lat2seed} x {self.lerpval}")
        lat1 = fromSeed(self.lat1seed)
        lat2 = fromSeed(self.lat2seed)
        lat = lat1 * (1 - self.lerpval) + lat2 * self.lerpval
        im = toImages(Gs, [toDLat(Gs, lat)], 300)[0]
        imqt = ImageQt(im)
        qtim = QImage(imqt)
        pixmap = QPixmap.fromImage(qtim)
        self.imlabel.setPixmap(pixmap)

    def createGif(self):
        name = f's{str(self.lat1seed)} to s{str(self.lat2seed)}.gif'
        os.makedirs(output_path, exist_ok=True)
        filename = os.path.join(output_path, name)
        print("Creating a gif file", filename)
        try:
            numFrames = 100
            fps = 16
            lat1 = fromSeed(self.lat1seed)
            lat2 = fromSeed(self.lat2seed)
            vals = [(math.sin(i) + 1) * 0.5 for i in np.linspace(0, 2 * math.pi, numFrames, False)]
            latents = np.array([lat1 * i + lat2 * (1 - i) for i in vals])
            frames = toImages(Gs, [toDLat(Gs, lat) for lat in latents], Gs.output_shape[2])
            frames[0].save(filename, save_all=True, append_images=frames[1:], duration=1000/fps, loop=0)
        except Exception as e:
            print("Error making gif...", e)
        else:
            print("Saved gif to ", filename)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    clock = Window()
    clock.show()
    sys.exit(app.exec_())