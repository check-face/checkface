"""Download NVIDIA's official StyleGAN2 FFHQ config-f network into the dnnlib cache and verify it.

checkface.py loads 'gdrive:networks/stylegan2-ffhq-config-f.pkl' through pretrained_networks.py, which maps it to
NVIDIA's official download URL and reads it from the same '.stylegan2-cache' directory this script fills, so the
API never downloads at runtime. The weights come straight from NVIDIA and are never redistributed by this repo.
"""
import glob
import hashlib
import os
import re
import sys

NETWORK = 'gdrive:networks/stylegan2-ffhq-config-f.pkl'
SHA256 = 'adf127ea7bb8a7788c8bdeda3c9937f7310b669b09ecf799ca53a631ff46948d'
SIZE = 381673535
CACHE_DIR = '.stylegan2-cache'


def network_url():
    # Read the URL from pretrained_networks.py rather than importing it: importing it needs TensorFlow, which the
    # lightweight download stage does not install. dnnlib's cache key is the md5 of exactly this URL.
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pretrained_networks.py')) as f:
        return re.search(r"'%s':\s*'([^']+)'" % re.escape(NETWORK), f.read()).group(1)


def cached_file(url):
    url_md5 = hashlib.md5(url.encode('utf-8')).hexdigest()
    cached = glob.glob(os.path.join(CACHE_DIR, url_md5 + '_*'))
    if len(cached) != 1:
        raise RuntimeError('Expected exactly one cached file for %s, found %r' % (url, cached))
    return cached[0]


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    import dnnlib
    url = network_url()
    dnnlib.util.open_url(url, cache_dir=CACHE_DIR).close()
    path = cached_file(url)
    digest = sha256_of(path)
    if os.path.getsize(path) != SIZE or digest != SHA256:
        os.remove(path)
        sys.exit('Downloaded network does not match the pinned official checkpoint (sha256 %s); removed it' % digest)
    print('Verified %s from %s (sha256 %s)' % (path, url, SHA256))


if __name__ == '__main__':
    main()
