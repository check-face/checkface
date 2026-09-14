"""Real legacy-route smoke against the running CPU/Mongo/ffmpeg application."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import uuid
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from bson.binary import Binary, UuidRepresentation
import numpy as np
from PIL import Image
import pymongo

BASE = os.getenv('CHECKFACE_TEST_URL', 'http://127.0.0.1:8080')
DATA = Path(os.getenv('CHECKFACE_DATA_DIR', '/app/checkfacedata'))
checks = []


def request(path, params=None, payload=None, headers=None, timeout=1200):
    if params:
        path += '?' + urlencode(params)
    body = None if payload is None else json.dumps(payload).encode()
    headers = {**({'Content-Type': 'application/json'} if body else {}), **(headers or {})}
    with urlopen(Request(BASE + path, data=body, headers=headers), timeout=timeout) as response:
        return response.read(), response.headers, response.status


def json_get(path, params=None):
    return json.loads(request(path, params)[0])


def image_get(path, params=None, size=None, format=None):
    body, headers, status = request(path, params)
    image = Image.open(io.BytesIO(body))
    image.load()
    if size:
        assert image.size == size, (path, image.size, size)
    if format:
        assert image.format == format, (path, image.format)
    return body, image, headers


assert json_get('/healthz')['ready']
assert request('/status/')[0] == b''
assert 'queue' in json_get('/api/queue/')
assert request('/api/encodeimage/')[2] == 200
assert b'<form' in request('/api/encodeimage/')[0]
checks.append('status/queue/upload form')

z0, z1 = [np.random.RandomState(seed).randn(512) for seed in (0, 1)]
for seed, expected in [(0, z0), (1, z1)]:
    response = json_get('/api/hashdata/', {'seed': seed})
    assert response['seed'] == seed
    np.testing.assert_array_equal(response['qlatent'], expected)
word = json_get('/api/hashdata/', {'value': 'hello'})
assert word['hash'] == hashlib.sha256(b'hello').hexdigest()
np.testing.assert_array_equal(word['qlatent'], np.random.RandomState(np.frombuffer(hashlib.sha256(b'hello').digest(), '<u4')).randn(512))
checks.append('deployed seed/text identity')

# Verify a GUID and its cached media survived the previous container start/run.
persisted = DATA / 'self-host-validation.json'
previous = {}
if persisted.exists():
    previous = json.loads(persisted.read_text())
    body = request('/api/face/', {'guid': previous['guid'], 'dim': 64})[0]
    assert hashlib.sha256(body).hexdigest() == previous['jpeg_sha256'], 'GUID/cache did not survive restart'
    checks.append('prior GUID/cache survives restart')

guid = request('/api/registerlatent/', payload={'latent': z0.tolist()})[0].decode()
uuid.UUID(guid)
np.testing.assert_array_equal(json_get('/api/hashdata/', {'guid': guid})['qlatent'], z0.astype(np.float32))
# Imported UUID binary documents keep working with the original test database.
client = pymongo.MongoClient(os.environ['MONGODB_CONNECTION_STRING'])
legacy_guid = uuid.uuid4()
legacy_id = Binary.from_uuid(legacy_guid, uuid_representation=UuidRepresentation.PYTHON_LEGACY)
client.test.latents.insert_one({'_id': legacy_id, 'latent': z1.tolist()})
np.testing.assert_array_equal(json_get('/api/hashdata/', {'guid': str(legacy_guid)})['qlatent'], z1)
checks.append('register/read GUID and imported legacy BSON UUID')

jpeg, face, headers = image_get('/api/face/', {'seed': 0, 'dim': 512}, (512, 512), 'JPEG')
reference = Image.open(Path(__file__).parent / 'fixtures/seed-0.png').convert('RGB')
encoded_reference = io.BytesIO()
reference.save(encoded_reference, format='JPEG')
expected = np.asarray(Image.open(io.BytesIO(encoded_reference.getvalue())), dtype=np.int16)
error = np.abs(np.asarray(face.convert('RGB'), dtype=np.int16) - expected)
assert error.mean() <= 0.25 and error.max() <= 12, ('JPEG versus fixed converted CPU reference', float(error.mean()), int(error.max()))
assert request('/api/face/', {'seed': 0, 'dim': 512})[0] == jpeg, 'Cached bytes changed'
try:
    request('/api/face/', {'seed': 0, 'dim': 512}, headers={'If-None-Match': headers['ETag']})
    raise AssertionError('Conditional image cache did not respond304')
except HTTPError as error_response:
    assert error_response.code == 304
image_get('/api/face/', {'seed': 0, 'dim': 64, 'format': 'webp'}, (64, 64), 'WEBP')
image_get('/api/face/', {'seed': 0, 'dim': 0}, (300, 300), 'JPEG')
image_get('/api/hello', size=(300, 300), format='JPEG')
guid_image = image_get('/api/face/', {'guid': guid, 'dim': 64}, (64, 64), 'JPEG')[0]
persisted.write_text(json.dumps({'guid': guid, 'jpeg_sha256': hashlib.sha256(guid_image).hexdigest()}) + '\n')
checks.append('JPEG/WebP/default/legacy image routes and conditional cache')

mixed_params = {'num_multi': 2, 'seed0': 0, 'seed1': 1, 'amount0': .25, 'amount1': .75}
np.testing.assert_allclose(json_get('/api/hashdata/', mixed_params)['qlatent'], z0*.25 + z1*.75, rtol=0, atol=1e-15)
image_get('/api/face/', {**mixed_params, 'dim': 64}, (64, 64), 'JPEG')
wplus = np.zeros((18, 512), dtype=np.float32)
w_guid = request('/api/registerlatent/', payload={'latent': wplus.tolist()})[0].decode()
np.testing.assert_array_equal(json_get('/api/hashdata/', {'guid': w_guid})['dlatent'], wplus)
image_get('/api/face/', {'guid': w_guid, 'dim': 64}, (64, 64), 'JPEG')
image_get('/api/morphframe/', {'from_guid': w_guid, 'to_seed': 0, 'linear': 'true', 'num_frames': 3, 'frame_num': 1, 'dim': 64}, (64, 64), 'JPEG')
image_get('/api/face/', {'num_multi': 2, 'guid0': w_guid, 'seed1': 0, 'amount0': .5, 'amount1': .5, 'dim': 64}, (64, 64), 'JPEG')
checks.append('weighted Z/mixed W+ blends and mixed-space interpolation')

morph = {'from_seed': 0, 'to_seed': 1, 'num_frames': 4, 'dim': 64, 'fps': 8}
frame = lambda index: request('/api/morphframe/', {**morph, 'frame_num': index})[0]
first, middle, last = frame(0), frame(1), frame(2)
assert frame(3) == middle, 'Trig mirrored frame cache differs'
assert first != middle and middle != last
for index, seed in [(0, 0), (2, 1)]:
    linear = request('/api/morphframe/', {'from_seed': 0, 'to_seed': 1, 'num_frames': 3, 'dim': 64, 'linear': 'true', 'frame_num': index})[0]
    seed_image = request('/api/face/', {'seed': seed, 'dim': 64})[0]
    assert linear == seed_image, 'Linear endpoint differs from requested face'
checks.append('classic trig deduplication and linear endpoint frames')

for route, fmt in [('gif', 'gif'), ('webp', 'webp'), ('mp4', 'mp4')]:
    body, headers, _ = request(f'/api/{route}/', morph)
    assert body == request(f'/api/{route}/', morph)[0], f'{route} cache changed'
    with tempfile.NamedTemporaryFile(suffix='.' + fmt) as output:
        output.write(body)
        output.flush()
        if fmt == 'webp':
            animated = Image.open(io.BytesIO(body))
            assert animated.size == (64, 64) and animated.n_frames == 4
        else:
            probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames', '-show_streams', '-of', 'json', output.name]))
            stream = probe['streams'][0]
            assert (stream['width'], stream['height']) == (64, 64)
            assert int(stream['nb_read_frames']) == 4, stream
    if route == 'mp4':
        partial, _, status = request('/api/mp4/', morph, headers={'Range': 'bytes=0-99'})
        assert status == 206 and partial == body[:100]
embedded = request('/api/mp4/', {**morph, 'embed_html': 'true'})[0]
assert b'data:video/mp4;base64,' in embedded and b'<video' in embedded
image_get('/api/linkpreview/', {'from_seed': 0, 'to_seed': 1, 'width': 600}, (600, 314), 'JPEG')
checks.append('GIF/animated WebP/MP4 codecs, cache, byte ranges, embed HTML and link preview')

def upload(image_bytes, try_align=True):
    boundary = 'checkface-test-' + uuid.uuid4().hex
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="tryalign"\r\n\r\n{str(try_align).lower()}\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="usrimg"; filename="synthetic.png"\r\n'
            'Content-Type: image/png\r\n\r\n').encode() + image_bytes + f'\r\n--{boundary}--\r\n'.encode()
    with urlopen(Request(BASE + '/api/encodeimage/', data=body,
                         headers={'Content-Type': 'multipart/form-data; boundary=' + boundary}), timeout=1200) as response:
        return json.loads(response.read())

fixture = (Path(__file__).parent / 'fixtures/seed-0.png').read_bytes()
encoded = upload(fixture, True)
assert encoded['did_align'] is True and upload(fixture, True) == encoded
if 'encoded_guid' in previous:
    assert encoded['guid'] == previous['encoded_guid'], 'Encoding cache GUID did not survive restart'
    checks.append('encoded upload cache survives restart')
encoded_latent = json_get('/api/hashdata/', {'guid': encoded['guid']})['dlatent']
expected_encoded = np.load(Path(__file__).parent / 'fixtures/seed-0-encoded-w-plus.npy', allow_pickle=False)
np.testing.assert_allclose(encoded_latent, expected_encoded, rtol=0, atol=0.0001)
image_get('/api/face/', {'guid': encoded['guid'], 'dim': 64}, (64, 64), 'JPEG')
unaligned = upload(fixture, False)
assert unaligned['did_align'] is False and unaligned['guid'] != encoded['guid']
assert upload(fixture, False) == unaligned
try:
    upload(b'not an image', True)
    raise AssertionError('Invalid uploaded image was accepted')
except HTTPError as response:
    assert response.code == 400
persisted.write_text(json.dumps({'guid': guid, 'jpeg_sha256': hashlib.sha256(guid_image).hexdigest(),
                                'encoded_guid': encoded['guid']}) + '\n')
checks.append('multipart encoding, aligned reference W+, reconstruction and aligned/unaligned GUID cache')

# Bounded admitted cache misses can wait on inference without consuming every HTTP worker.
# Unique seeds avoid turning this into an accidental cache-hit-only health test.
capacity = json_get('/healthz')['generation_capacity']
seeds = [int.from_bytes(os.urandom(4), 'little') for _ in range(capacity)]
with ThreadPoolExecutor(max_workers=capacity) as pool:
    pending = [pool.submit(request, '/api/face/', {'seed': seed, 'dim': 32}) for seed in seeds]
    deadline = time.monotonic() + 30
    while json_get('/api/queue/')['queue'] < capacity - 1:
        assert time.monotonic() < deadline and not all(item.done() for item in pending), 'Did not exercise queued inference'
        time.sleep(.01)
    start = time.monotonic()
    assert json_get('/healthz')['ready'] and request('/status/', timeout=2)[0] == b''
    assert time.monotonic() - start < 2
    try:
        request('/api/face/', {'seed': 12345, 'dim': 32}, timeout=2)
        raise AssertionError('Generation beyond capacity should be503')
    except HTTPError as response:
        assert response.code == 503 and response.headers['Retry-After'] == '1'
    for item in pending:
        assert item.result(timeout=1200)[2] == 200
assert json_get('/api/queue/')['queue'] == 0
checks.append('responsive health under queued inference, bounded admission and recovery')

for payload in ({}, {'latent': [1.0]}, {'latent': ['bad']*512}):
    try:
        request('/api/registerlatent/', payload=payload)
        raise AssertionError('Invalid registration accepted')
    except HTTPError as response:
        assert response.code == 400
print(json.dumps({'result': 'pass', 'checks': checks, 'reference_jpeg_max_error': int(error.max()), 'reference_jpeg_mean_error': float(error.mean()), 'guid': guid}))
