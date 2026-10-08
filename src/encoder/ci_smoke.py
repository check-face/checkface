"""Real encode through the HTTP route (works on CPU, used by CI): python ci_smoke.py <face-image> [out.npy]

Starts the worker exactly as server_inference.main() does (ENCODER_DEVICE, default auto) and posts the image with
tryalign true and false, checking the response contract the API relies on: 18x512 finite floats + did_align.
"""
import os
import sys
import threading

import numpy as np

import server_inference as s

image = sys.argv[1]
s.device = s.pick_device(os.environ.get('ENCODER_DEVICE', 'auto'))
s.dlib_shape_predictor = s.dlib.shape_predictor(os.path.join(s.MODELS, 'shape_predictor_68_face_landmarks.dat'))
threading.Thread(target=s.worker, args=[os.path.join(s.MODELS, 'e4e_ffhq_encode.pt'), 1], daemon=True).start()
client = s.app.test_client()
results = {}
for align in ('true', 'false'):
    with open(image, 'rb') as f:
        r = client.post('/api/encodeimage/', data={'usrimg': (f, 'face.jpg'), 'tryalign': align})
    assert r.status_code == 200, r.data
    body = r.get_json()
    latent = np.array(body['dlatent'], dtype=np.float32)
    assert latent.shape == (18, 512) and np.isfinite(latent).all(), latent.shape
    assert body['did_align'] is (align == 'true'), body['did_align']
    results[align] = latent
    print('ok encode tryalign=%s on %s: did_align=%s mean=%.4f std=%.4f' % (align, s.device, body['did_align'], latent.mean(), latent.std()))
r = s.app.test_client().post('/api/encodeimage/', data={'tryalign': 'true'})
assert r.status_code == 400
print('ok missing upload -> 400')
if len(sys.argv) > 2:
    np.save(sys.argv[2], np.stack([results['true'], results['false']]))
