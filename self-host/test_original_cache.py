"""Real PNG storage/recovery checks, without model weights or inference."""
from concurrent.futures import ThreadPoolExecutor
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from PIL import Image
from original_cache import OriginalCache


class OriginalCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = OriginalCache(self.root)
        self.calls = 0
        self.identity = {'model': 'fixture-model', 'latent': 'fixture-w-plus', 'noise': 'constant'}

    def generate(self):
        self.calls += 1
        # Non-uniform pixels catch accidental resize, conversion and lossy storage.
        return Image.frombytes('RGB', (1024, 1024), bytes(range(256)) * (1024 * 1024 * 3 // 256))

    def test_sizes_formats_and_restart_reuse_raw_pixels(self):
        original = self.cache.get_or_create(self.identity, self.generate)
        original.resize((256, 256)).save(self.root / 'derived.jpg', format='JPEG')
        restarted = OriginalCache(self.root)
        recovered = restarted.get_or_create(self.identity, self.generate)
        recovered.resize((512, 512)).save(self.root / 'derived.webp', format='WEBP')
        self.assertEqual(self.calls, 1)
        self.assertEqual(recovered.tobytes(), original.tobytes())
        self.assertEqual(recovered.size, (1024, 1024))

    def test_separate_instances_deduplicate_concurrent_requests(self):
        caches = [OriginalCache(self.root) for _ in range(8)]
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda cache: cache.get_or_create(self.identity, self.generate), caches))
        self.assertEqual(self.calls, 1)
        self.assertTrue(all(image.tobytes() == results[0].tobytes() for image in results))

    def test_changed_identity_regenerates(self):
        self.cache.get_or_create(self.identity, self.generate)
        for field in ['model', 'latent', 'noise']:
            self.cache.get_or_create(dict(self.identity, **{field: 'changed'}), self.generate)
        self.assertEqual(self.calls, 4)

    def test_historic_derivative_never_counts_as_original(self):
        legacy = self.root / 'outputImages'
        legacy.mkdir()
        Image.new('RGB', (300, 300)).save(legacy / 's0_300.jpg')
        before = (legacy / 's0_300.jpg').read_bytes()
        self.cache.get_or_create(self.identity, self.generate)
        self.assertEqual(self.calls, 1)
        self.assertEqual((legacy / 's0_300.jpg').read_bytes(), before)

    def test_corrupt_and_partial_files_are_not_hits(self):
        self.cache.get_or_create(self.identity, self.generate)
        next(self.cache.root.glob('*.png')).write_bytes(b'broken')
        pending = self.cache.root / '.pending-interrupted'
        pending.write_bytes(b'partial')
        self.cache.get_or_create(self.identity, self.generate)
        self.assertEqual(self.calls, 2)
        self.assertFalse(pending.exists())

    def test_wrong_pixels_with_old_digest_regenerate(self):
        self.cache.get_or_create(self.identity, self.generate)
        path = next(self.cache.root.glob('*.png'))
        Image.new('RGB', (1024, 1024)).save(path)
        self.cache.get_or_create(self.identity, self.generate)
        self.assertEqual(self.calls, 2)

    def test_quota_returns_result_without_deleting_originals(self):
        self.cache.get_or_create(self.identity, self.generate)
        before = next(self.cache.root.glob('*.png')).read_bytes()
        full = OriginalCache(self.root, max_bytes=0)
        result = full.get_or_create({'new': True}, self.generate)
        self.assertEqual(result.size, (1024, 1024))
        self.assertEqual(full.write_failures, 1)
        self.assertEqual(next(self.cache.root.glob('*.png')).read_bytes(), before)
        self.assertFalse(list(self.cache.root.glob('.pending-*')))

    def test_atomic_promotion_failure_recovers(self):
        with patch('original_cache.os.replace', side_effect=OSError('disk full')):
            self.cache.get_or_create(self.identity, self.generate)
        self.assertFalse(list(self.cache.root.glob('*.png')))
        self.assertFalse(list(self.cache.root.glob('.pending-*')))
        self.cache.get_or_create(self.identity, self.generate)
        self.assertEqual(self.calls, 2)

    def test_generation_error_is_not_retried_or_cached(self):
        def fail():
            self.calls += 1
            raise OSError('inference failed')
        with self.assertRaisesRegex(OSError, 'inference failed'):
            self.cache.get_or_create(self.identity, fail)
        self.assertEqual(self.calls, 1)
        self.assertFalse(list(self.cache.root.glob('*.png')))

    def test_invalid_output_not_cached(self):
        with self.assertRaisesRegex(ValueError, '1024'):
            self.cache.get_or_create(self.identity, lambda: Image.new('RGB', (256, 256)))
        self.assertFalse(list(self.cache.root.glob('*.png')))


if __name__ == '__main__':
    unittest.main()
