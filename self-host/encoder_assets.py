"""Exact deployed e4e asset identities, shared by setup and inference."""
import hashlib
import os
from pathlib import Path

ASSETS = {
    'e4e_ffhq_encode.pt': {
        'sha256': '2ace1d9a8c05c10a399bcd500b8dda118f759ff1aac89dbdab7435f2136a0999',
        'bytes': 1201649680,
        'url': 'https://drive.usercontent.google.com/download?id=1cUv_reLE6k3604or78EranS7XzuVMWeO&export=download&confirm=t',
        'source': 'https://github.com/omertov/encoder4editing#pretrained-models',
    },
    'shape_predictor_68_face_landmarks.dat': {
        'sha256': 'fbdc2cb80eb9aa7a758672cbfdda32ba6300efe9b6e6c7a299ff7e736b11b92f',
        'bytes': 99693937,
        'url': 'https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2',
        'source': 'https://dlib.net/face_landmark_detection.py.html',
        'compression': 'bz2',
    },
}

def model_directory():
    return Path(os.environ.get('CHECKFACE_MODEL_DIR', '/models'))

def verified_asset(name, directory=None):
    entry = ASSETS[name]
    path = (directory or model_directory()) / name
    if path.stat().st_size != entry['bytes']:
        raise ValueError(f'Unverified encoder asset size: {name}; run prepare_encoder.py')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != entry['sha256']:
        raise ValueError(f'Unverified encoder asset checksum: {name}; no model loaded')
    return path

def verify_assets(directory=None):
    return {name: {'sha256': ASSETS[name]['sha256'], 'bytes': verified_asset(name, directory).stat().st_size} for name in ASSETS}
