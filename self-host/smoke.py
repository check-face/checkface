"""Real-model HTTP smoke; never substitutes mock inference or skips missing assets."""
import hashlib
from concurrent.futures import ThreadPoolExecutor
import time
import io
import json
import sys
from pathlib import Path
import numpy as np
import os
import subprocess
import tempfile
from prepare_model import MODEL_NAME, SOURCE_SHA256
from urllib.error import HTTPError
from urllib.request import urlopen
import zipfile
from PIL import Image

base = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8080'
def get(path, timeout=600):
    with urlopen(base + path, timeout=timeout) as response:
        return response.read()
def pixels(data):
    image = Image.open(io.BytesIO(data))
    assert image.size == (1024, 1024) and image.mode == 'RGB'
    assert max(hi - lo for lo, hi in image.getextrema()) > 32, 'Blank output'
    return image.tobytes()

health = json.loads(get('/healthz'))
assert health['provider'] == 'cpu' and health['model_source_sha256'] == SOURCE_SHA256
model_directory = Path(os.environ.get('CHECKFACE_MODEL', '/models/' + MODEL_NAME)).parent
provenance_path = model_directory / 'provenance.json'
if provenance_path.exists():
    assert json.loads(provenance_path.read_text())['source_sha256'] == SOURCE_SHA256
with tempfile.TemporaryDirectory() as temporary:
    wrong_model = Path(temporary) / 'wrong.pkl'
    wrong_model.write_bytes(b'not a trusted pickle')
    startup = subprocess.run([sys.executable, str(Path(__file__).with_name('api.py'))],
        env={**os.environ, 'CHECKFACE_MODEL': str(wrong_model)}, capture_output=True, text=True, timeout=60)
    assert startup.returncode != 0 and 'Untrusted or wrong model' in startup.stderr, startup.stderr
a = pixels(get('/v1/face?seed=0'))
assert a == pixels(get('/v1/face?seed=0')), 'Seed inference is not deterministic'
# Real inference remains bounded while health requests run independently.
with ThreadPoolExecutor(max_workers=1) as pool:
    ongoing = pool.submit(get, '/v1/face?seed=0')
    deadline = time.monotonic() + 30
    while True:
        busy = json.loads(get('/healthz', timeout=2))['inference_busy']
        if busy:
            break
        assert not ongoing.done(), 'Inference completed before admission could be tested'
        assert time.monotonic() < deadline, 'Inference never entered busy state'
        time.sleep(0.01)
    started = time.monotonic()
    assert json.loads(get('/healthz', timeout=2))['ready']
    try:
        get('/v1/face?seed=1', timeout=2)
        raise AssertionError('Concurrent inference was accepted instead of rejected')
    except HTTPError as error:
        assert error.code == 503 and error.headers['Retry-After'] == '1'
    assert time.monotonic() - started < 2, 'Health/admission waited for inference'
    assert pixels(ongoing.result(timeout=600)) == a
assert not json.loads(get('/healthz', timeout=2))['inference_busy']
b = pixels(get('/v1/face?seed=1'))
assert a != b, 'Distinct seeds produced identical images'
reference_errors = {}
for seed, data in [(0, a), (1, b)]:
    actual = Image.frombytes('RGB', (1024, 1024), data).resize((512, 512), Image.Resampling.LANCZOS)
    expected = Image.open(Path(__file__).parent / 'fixtures' / f'seed-{seed}.png').convert('RGB')
    error = int(np.abs(np.asarray(actual, dtype=np.int16) - np.asarray(expected, dtype=np.int16)).max())
    assert error <= 1, f'Seed {seed} differs from independent converted CPU reference: max error {error}'
    reference_errors[str(seed)] = error
word = pixels(get('/v1/face?text=hello'))
assert word != a and word == pixels(get('/v1/face?text=hello')), 'Word identity/repeatability failed'
with zipfile.ZipFile(io.BytesIO(get('/v1/morph?from=0&to=1&frames=3'))) as archive:
    assert pixels(archive.read('000.png')) == a
    assert pixels(archive.read('002.png')) == b
    assert pixels(archive.read('001.png')) not in (a, b)
    assert json.loads(archive.read('manifest.json'))['frames'] == 3
for path in ['/v1/face', '/v1/face?seed=-1', '/v1/face?seed=4294967296', '/v1/face?seed=1&seed=2', '/v1/face?seed=0&text=hello', '/v1/morph?from=0&to=1&frames=1000']:
    try:
        get(path)
        raise AssertionError('Invalid request accepted')
    except HTTPError as error:
        assert error.code == 400
try:
    get('/api/encodeimage')
    raise AssertionError('Unsupported legacy route was accepted')
except HTTPError as error:
    assert error.code == 404
print(json.dumps({'result': 'pass', 'seed0_rgb_sha256': hashlib.sha256(a).hexdigest(), 'fixed_reference_max_errors': reference_errors,
                  'checks': ['1024px real inference', 'repeatability', 'fixed CPU references', 'word identity', 'distinct seeds', 'morph endpoints/midpoint', 'input bounds', 'wrong model rejection', 'unsupported routes', 'health during inference', 'busy admission503']}))
