"""Migration contract checks: defaults, frame geometry, mixed Z/W+, paths and routes."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import os
import queue
import tempfile
from unittest.mock import patch
import numpy as np
import legacy_checkface as api


class FixedLatent(api.LatentProxy):
    def __init__(self, value, name):
        self.value, self.name = np.array(value), name
    def getLatent(self, _Gs=None):
        return self.value
    def getName(self):
        return self.name
    def getShardPartitions(self):
        return ['fixture']


class LegacyContract(unittest.TestCase):
    def test_deployed_routes_retained(self):
        inventory = json.loads(Path(__file__).with_name('legacy-route-inventory.json').read_text())
        rules = {rule.rule: rule.methods for rule in api.app.url_map.iter_rules()}
        for route in inventory['routes']:
            self.assertTrue(set(route['methods']).issubset(rules[route['path']]), route)

    def test_request_defaults_and_identity_precedence(self):
        for query in ['', '?dim=bad', '?dim=9', '?dim=1025']:
            with api.app.test_request_context('/api/face/' + query):
                self.assertEqual(api.getRequestedImageDim(api.request), 300)
        with api.app.test_request_context('/api/face/?dim=512&value=ignored&seed=7'):
            self.assertEqual(api.getRequestedImageDim(api.request), 512)
            self.assertEqual(api.getRequestLatent(api.request).getName(), 's7')
        with api.app.test_request_context('/api/morphframe/?linear=TrUe'):
            self.assertTrue(api.argIsTrue(api.request, 'linear'))
        self.assertEqual(api.getFramesMorphdir('/cache', 50, 300, False), '/cache/frames/trig n50x300')
        self.assertEqual(api.getFramesMorphdir('/cache', 3, 64, True), '/cache/frames/linear n3x64')

    def test_word_seed_and_cache_identity(self):
        seed = api.LatentBySeed(42)
        np.testing.assert_array_equal(seed.getLatent(), np.random.RandomState(42).randn(512))
        self.assertEqual(seed.getShardPartitions(), ['s42', '42'])
        word = api.LatentByTextValue('hello')
        digest = hashlib.sha256(b'hello').digest()
        np.testing.assert_array_equal(word.getLatent(), np.random.RandomState(np.frombuffer(digest, '<u4')).randn(512))
        self.assertEqual(word.getName(), 'hash-' + digest.hex())
        self.assertEqual(api.encodeRequestKey(b'fixture', True), hashlib.sha256(b'fixture').hexdigest() + '-tryalign=True')

    def test_mixed_interpolation_and_existing_multi_weight_order(self):
        z = np.arange(512, dtype=np.float32) / 512
        w = np.ones((18, 512), dtype=np.float32) * 3
        mapping = lambda values, _labels: np.repeat(np.asarray(values)[:, None, :], 18, axis=1)
        model = SimpleNamespace(components=SimpleNamespace(mapping=SimpleNamespace(run=mapping)))
        with patch.object(api, 'dlatent_avg', np.zeros(512, dtype=np.float32)):
            q, d = FixedLatent(z, 'q'), FixedLatent(w, 'd')
            mapped = api.toDLat(model, z)
            np.testing.assert_allclose(api.LatentByLerp(q, d, .25).getLatent(model), mapped*.75 + w*.25)
            # Preserve the deployed algorithm: amounts are applied before converting
            # Z to W+ in mixed multi blends; do not silently substitute a new policy.
            mixed = api.LatentByMultiLerp([[.25, FixedLatent(z, 'q')], [.75, FixedLatent(w, 'd')]])
            np.testing.assert_allclose(mixed.getLatent(model), api.toDLat(model, z*.25) + w*.75)
            straight = api.LatentByMultiLerp([[.25, FixedLatent(z, 'q')], [.75, FixedLatent(z*2, 'q2')]])
            np.testing.assert_allclose(straight.getLatent(model), z*1.75)

    def test_original_oembed_contract(self):
        with api.app.test_client() as client:
            self.assertEqual(client.get('/oembed.json').status_code, 400)
            self.assertEqual(client.get('/oembed.json?format=xml&url=/').status_code, 501)
            self.assertEqual(client.get('/oembed.json?url=/&maxwidth=49').status_code, 400)
            single = client.get('/oembed.json', query_string={'url': '/?from_value=hello'}).json
            self.assertEqual((single['type'], single['width'], single['title']), ('photo', 512, 'hello'))
            self.assertIn('/api/face/?dim=512&value=hello&format=webp', single['url'])
            movie = client.get('/oembed.json/', query_string={'url': '/?from_seed=0&to_seed=1', 'maxwidth': 200, 'maxheight': 300}).json
            self.assertEqual((movie['width'], movie['height'], movie['title']), (200, 200, 'seed 0 to seed 1'))
            self.assertIn('/api/webp/?dim=200&from_seed=0&to_seed=1', movie['url'])
            self.assertEqual(movie['thumbnail_width'], 128)

    def test_full_queue_has_bounded_error_and_recovers(self):
        limited = queue.Queue(maxsize=1)
        limited.put(object())
        with tempfile.TemporaryDirectory() as directory, patch.object(api, 'q', limited), \
             patch.object(api, 'DATA_DIR', Path(directory)), \
             patch.dict(os.environ, {'CHECKFACE_ENQUEUE_TIMEOUT': '0.01'}):
            with api.app.test_client() as client:
                response = client.get('/api/face/?seed=314159&dim=10')
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.headers['Retry-After'], '1')
                self.assertEqual(client.get('/api/queue/').json['queue'], 1)
                self.assertEqual(client.get('/status/').status_code, 200)
            limited.get_nowait()
            api.enqueue(api.GenerateImageJob(api.LatentBySeed(0), 'recovered'))
            self.assertEqual(limited.qsize(), 1)

    def test_invalid_latents_rejected_before_persistence(self):
        for value in ([1.0], [float('nan')]*512, [['bad']]):
            self.assertFalse(api.registerLatent(value)[0])
        with api.app.test_client() as client:
            self.assertEqual(client.post('/api/registerlatent/', json={}).status_code, 400)
            self.assertEqual(client.post('/api/encodeimage/', data={}).status_code, 400)
            self.assertEqual(client.get('/api/encodeimage/').status_code, 200)
            self.assertEqual(client.get('/status/').data, b'')
            self.assertEqual(client.get('/api/queue/').json, {'queue': 0})


if __name__ == '__main__':
    unittest.main()
