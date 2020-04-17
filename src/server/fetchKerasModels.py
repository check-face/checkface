# load face landmarks detector
import bz2
from keras.utils import get_file

LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'

import os
def unpack_bz2(src_path):
    dst_path = src_path[:-4]
    if not os.path.exists(dst_path):
      data = bz2.BZ2File(src_path).read()
      with open(dst_path, 'wb') as fp:
          fp.write(data)
    return dst_path

landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                        LANDMARKS_MODEL_URL, cache_subdir='temp'))

# load VGG16 for perceptual model
from keras.applications.vgg16 import VGG16
VGG16(include_top=False)