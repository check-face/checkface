#!/usr/bin/env python3

import flask
import threading
import subprocess
import shutil
import base64
import math
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

output_path = r"C:\Users\Oliver\Source\Repos\stylegan\results\outgifs"
# def fetch_model():
#     snap = r"C:\Users\Oliver\Source\Repos\stylegan\results\00007-sgan-flower-1gpu\network-snapshot-008190.pkl"
#     #snap = r"C:\Users\Oliver\Downloads\network-snapshot-010461\network-snapshot-010461.pkl"

#     with open(snap, 'rb') as f:
#         _G, _D, Gs = pickle.load(f)
#         return Gs

def fetch_model():
    url = 'https://drive.google.com/uc?id=1-O8VHNOpBNHnQyn0yz_pK3PHoc3CboC3'
    import dnnlib
    with dnnlib.util.open_url(url, cache_dir='cache') as f:
        _G, _D, Gs = pickle.load(f)
        return Gs



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

    #print(f"Took {diff:.2f} seconds to run {network}")
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
from PyQt5.QtCore import Qt, QTimer, QEvent, pyqtSignal, QObject
from PyQt5.QtWidgets import (QApplication, QCheckBox, QGridLayout, QGroupBox, QLabel,
                             QMenu, QPushButton, QRadioButton, QVBoxLayout, QWidget, QSlider, QLineEdit)
from PyQt5.QtGui import QPixmap, QImage
from PIL.ImageQt import ImageQt

Gsmodel = None
synthesis_layers = None
dlatent_avg = None
synthesis_kwargs = None

def Gs():
    global Gsmodel
    global synthesis_layers
    global dlatent_avg
    global synthesis_kwargs
    if Gsmodel:
        return Gsmodel
    else:
        import dnnlib.tflib as tflib
        import dnnlib
        dnnlib.tflib.init_tf()
        Gsmodel = fetch_model()
        synthesis_layers = Gsmodel.components.synthesis.input_shape[1]
        dlatent_avg = Gsmodel.get_var('dlatent_avg')
        synthesis_kwargs = dict(output_transform=dict(
            func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=20)
        return Gsmodel

class FocusChangedEventFilter(QObject):
    focusChanged = pyqtSignal(bool)
    def __init__(self, parent = None):
        super(FocusChangedEventFilter, self).__init__(parent)

    def eventFilter(self, object, event):
        if event.type()== QEvent.FocusIn:
            self.focusChanged.emit(True)
        elif event.type()== QEvent.FocusOut:
            self.focusChanged.emit(False)
        return False

class Window(QWidget):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        grid = QGridLayout()
        self.imresolution = 512
        self.imlabel = QLabel()
        self.imlabel.setFixedWidth(self.imresolution)
        self.imlabel.setFixedHeight(self.imresolution)
        self.imlabel.setScaledContents(True)
        self.lat1seed = 13
        self.lat2seed = 5
        self.lerpval = 0
        self.prevlat = fromSeed(1)
        self.prevlat2seed = None
        self.prevlat1seed = None
        grid.addWidget(self.imlabel, 0, 0)

        slider1 = self.createSlider()
        slider1.valueChanged.connect(self.slider1changed)
        slider1.sliderReleased.connect(self.sliderReleased)
        grid.addWidget(slider1, 1, 0)

        slider2 = self.createSlider()
        slider2.valueChanged.connect(self.slider2changed)
        slider2.sliderReleased.connect(self.sliderReleased)
        grid.addWidget(slider2, 2, 0)

        slider3 = self.createSlider()
        slider3.valueChanged.connect(self.slider3changed)
        slider3.sliderReleased.connect(self.sliderReleased)
        grid.addWidget(slider3, 3, 0)
        
        button = QPushButton('Create gif')
        button.clicked.connect(self.createGif)
        grid.addWidget(button, 4, 0)

        self.textbox = QLineEdit("hello world")
        self.textbox.textChanged.connect(self.textboxChanged)
        fcev = FocusChangedEventFilter(self)
        self.textbox.installEventFilter(fcev)
        fcev.focusChanged.connect(self.textboxFocusChanged)
        grid.addWidget(self.textbox, 5, 0)

        
        self.setLayout(grid)

        self.setWindowTitle("Checkface Sliders")
        #self.resize(400, 300)

        self.animTheta = 0
        self.animThetaPause = 0
        self.enableAnim = True
        timer = QTimer(self)
        timer.timeout.connect(self.tick)
        timer.start(75)


    def createSlider(self):
        slider = QSlider(Qt.Horizontal)
        slider.setFocusPolicy(Qt.StrongFocus)
        slider.setTickPosition(QSlider.TicksBothSides)
        slider.setTickInterval(10)
        slider.setSingleStep(1)
        return slider

    def sliderReleased(self):
        self.enableAnim = True

    def slider1changed(self, val):
        self.lat1seed = val
        self.lerpval = 0
        self.enableAnim = False
        self.animTheta = 0
        self.renderImage()


    def slider2changed(self, val):
        self.lat2seed = val
        self.lerpval = 1
        self.animTheta = math.pi
        self.enableAnim = False
        self.renderImage()


    def slider3changed(self, val):
        self.lerpval = val / 100
        self.animTheta = math.acos(self.lerpval * 2 - 1) + math.pi
        self.enableAnim = False
        self.renderImage()


    def makeGrid(self, image):
        img_w, img_h = image.size
        background = PIL.Image.new('RGBA',(img_w * 3, img_h * 3), (255, 255, 255, 255))
        bg_w, bg_h = background.size
        background.paste(image,(0, 0))
        background.paste(image,(img_w, 0))
        background.paste(image,(img_w * 2, 0))
        background.paste(image,(0, img_h))
        background.paste(image,(img_w, img_h))
        background.paste(image,(img_w * 2, img_h))
        background.paste(image,(0, img_h * 2))
        background.paste(image,(img_w, img_h * 2))
        background.paste(image,(img_w * 2, img_h * 2))
        return background

    def showImage(self, im):
        imqt = ImageQt(im)
        qtim = QImage(imqt)
        pixmap = QPixmap.fromImage(qtim)
        self.imlabel.setPixmap(pixmap)

        # Avoid garbage collecting QImage until has been replaced by next one
        # Another solution would be to use pixmap.copy above - see https://stackoverflow.com/questions/40356441/python3-pillow-qt5-crash-when-i-resize-a-label-containing-an-image
        self.previmqt = imqt

    def renderImage(self):
        #print(f"s{self.lat1seed} to s{self.lat2seed} x {self.lerpval}")
        lat1 = fromSeed(self.lat1seed)
        lat2 = fromSeed(self.lat2seed)
        lat = lat1 * (1 - self.lerpval) + lat2 * self.lerpval
        if np.array_equal(lat, self.prevlat):
            return

        self.prevlat = lat
        im = toImages(Gs(), [toDLat(Gs(), lat)], self.imresolution)[0]
        #im = PIL.Image.open(r"C:\Users\Oliver\Documents\StyleGanFaces\s705970_300.jpg")
        #wd = 512
        #im = im.resize((wd, wd), PIL.Image.ANTIALIAS)
        self.showImage(im)

    # def renderCheckImage(self):
    #     lat = fromSeed(hashToSeed(self.textbox.text()))
    #     im = toImages(Gs(), [toDLat(Gs(), lat)], self.imresolution)[0]
    #     self.showImage(im)

    def useCheckImages(self):
        strs = self.textbox.text().split('-', 1)
        self.lat1seed = hashToSeed(strs[0])
        if len(strs) > 1:
            self.lat2seed = hashToSeed(strs[1])
        else:
            self.lat2seed = self.lat1seed

        if self.lat1seed != self.prevlat1seed:
            self.lerpval = 0
            self.animTheta = 0
        elif self.lat2seed != self.prevlat2seed:
            self.lerpval = 1
            self.animTheta = math.pi

        self.prevlat1seed = self.lat1seed
        self.prevlat2seed = self.lat2seed
        self.animThetaPause = math.pi * 2 / 6

        self.renderImage()


    def textboxChanged(self, value):
        self.useCheckImages()
        # self.renderCheckImage()

    def textboxFocusChanged(self, isFocussed):
        if isFocussed:
            self.useCheckImages()

        # if isFocussed:
        #     self.enableAnim = False
        #     self.renderCheckImage()
        # else:
        #     self.enableAnim = True
        #     self.renderImage()



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
            frames = toImages(Gs(), [toDLat(Gs(), lat) for lat in latents], Gs().output_shape[2])
            frames[0].save(filename, save_all=True, append_images=frames[1:], duration=1000/fps, loop=0)
        except Exception as e:
            print("Error making gif...", e)
        else:
            print("Saved gif to ", filename)

    def tick(self):
        if self.enableAnim:
            self.lerpval = (math.cos(self.animTheta + math.pi) + 1) * 0.5
            if self.animThetaPause > 0:
                self.animThetaPause = self.animThetaPause - math.pi * 2 / 100
            else:
                self.animTheta = self.animTheta + math.pi * 2 / 100
            self.renderImage()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    clock = Window()
    clock.show()
    sys.exit(app.exec_())