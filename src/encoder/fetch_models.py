"""Download the encoder's model files from their original publishers and verify pinned sizes and sha256.

fetch_models.py <directory> [name ...]   (default: all)
  e4e_ffhq_encode.pt                     encoder4editing FFHQ encoder (link in https://github.com/omertov/encoder4editing)
  shape_predictor_68_face_landmarks.dat  dlib face landmarks (https://dlib.net/files/), used for face alignment
Neither file is redistributed by this repository. See SELF-HOSTING.md for their terms.
"""
import bz2
import hashlib
import os
import sys
import time
import urllib.request

ASSETS = {
    'e4e_ffhq_encode.pt': {
        'url': 'https://drive.usercontent.google.com/download?id=1cUv_reLE6k3604or78EranS7XzuVMWeO&export=download&confirm=t',
        'sha256': '2ace1d9a8c05c10a399bcd500b8dda118f759ff1aac89dbdab7435f2136a0999',
        'bytes': 1201649680,
    },
    'shape_predictor_68_face_landmarks.dat': {
        'url': 'https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2',
        'sha256': 'fbdc2cb80eb9aa7a758672cbfdda32ba6300efe9b6e6c7a299ff7e736b11b92f',
        'bytes': 99693937,
        'bz2': True,
    },
}



def download(entry, temporary):
    digest, size = hashlib.sha256(), 0
    with urllib.request.urlopen(entry['url'], timeout=300) as response, open(temporary, 'wb') as out:
        source = bz2.BZ2File(response) if entry.get('bz2') else response
        while True:
            chunk = source.read(1 << 20)
            if not chunk:
                break
            size += len(chunk)
            if size > entry['bytes']:
                break
            digest.update(chunk)
            out.write(chunk)
    return size, digest.hexdigest()


def fetch(name, entry, directory, attempts=4):
    target = os.path.join(directory, name)
    temporary = target + '.download'
    for attempt in range(1, attempts + 1):
        try:
            size, digest = download(entry, temporary)
        except (OSError, EOFError) as error:  # dropped or truncated connection
            print('%s: attempt %d failed: %r' % (name, attempt, error), flush=True)
            time.sleep(10 * attempt)
            continue
        if size == entry['bytes'] and digest == entry['sha256']:
            os.replace(temporary, target)
            print('Verified %s (sha256 %s)' % (target, entry['sha256']), flush=True)
            return
        print('%s: attempt %d does not match the pinned file (%d bytes, sha256 %s)' % (name, attempt, size, digest), flush=True)
    if os.path.exists(temporary):
        os.remove(temporary)
    sys.exit('%s: could not download the pinned file from %s' % (name, entry['url']))


if __name__ == '__main__':
    directory = sys.argv[1]
    os.makedirs(directory, exist_ok=True)
    for name in sys.argv[2:] or list(ASSETS):
        fetch(name, ASSETS[name], directory)
