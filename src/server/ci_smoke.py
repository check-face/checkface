"""GPU-free checks of a built API image (used by CI): python ci_smoke.py <sm_XX>

1. The baked StyleGAN2 checkpoint is NVIDIA's pinned file, cached where checkface.py will load it (no runtime download).
2. Both StyleGAN2 custom CUDA ops compile with the image's nvcc for <sm_XX> and load into TensorFlow. On a GPU host
   the arch comes from the GPU; CI runners have none, so it is passed in.
3. checkface.py imports and serves the routes that do not need the generator, per docs/api.md.
Generation itself needs an NVIDIA GPU; SELF-HOSTING.md has the on-GPU verification commands.
"""
import hashlib
import os
import sys

import numpy as np

arch = sys.argv[1]
os.environ.setdefault('MONGODB_CONNECTION_STRING', 'mongodb://127.0.0.1:1/?serverSelectionTimeoutMS=100')

import fetchModel  # noqa: E402

path = fetchModel.cached_file(fetchModel.network_url())
assert os.path.getsize(path) == fetchModel.SIZE and fetchModel.sha256_of(path) == fetchModel.SHA256, path
import pretrained_networks  # noqa: E402
assert pretrained_networks.get_path_or_url(fetchModel.NETWORK) == fetchModel.network_url()
print('ok checkpoint', path)

import dnnlib.tflib.custom_ops as custom_ops  # noqa: E402
from dnnlib.tflib.ops import fused_bias_act, upfirdn_2d  # noqa: E402

custom_ops._get_cuda_gpu_arch_string = lambda: arch
for op in (fused_bias_act, upfirdn_2d):
    plugin = op._get_plugin()
    print('ok custom op', op.__name__, [name for name in dir(plugin) if not name.startswith('_')][:3])

import checkface  # noqa: E402

client = checkface.app.test_client()
assert client.get('/status/').status_code == 200
assert client.get('/').data == b'It works'
assert client.get('/api/queue/').get_json() == {'queue': 0}
data = client.get('/api/hashdata/?seed=1').get_json()
assert data['seed'] == 1 and np.allclose(data['qlatent'], np.random.RandomState(1).randn(512))
data = client.get('/api/hashdata/?value=example').get_json()
assert data['hash'] == hashlib.sha256(b'example').hexdigest() and len(data['qlatent']) == 512
data = client.get('/api/hashdata/?num_multi=2&seed0=1&seed1=2').get_json()
assert np.allclose(data['qlatent'], 0.5 * np.random.RandomState(1).randn(512) + 0.5 * np.random.RandomState(2).randn(512))
print('ok routes')
