import hashlib
from pathlib import Path
import tempfile
import unittest
from cache_inventory import classify, scan


class InventoryTests(unittest.TestCase):
    def test_strict_seed_shards(self):
        self.assertEqual(classify('outputImages/s1/101/s101_300.jpg')[0], 'seed_path_candidate')
        for path in ['outputImages/s1/101/s102_300.jpg', 'outputImages/s1/101/s101_1.jpg', 'outputImages/s1/101/s101_300.jpg.tmp', 'outputImages/s1/101/hash-123_300.jpg', 'outputImages/s0/0/s4294967296_300.jpg']:
            self.assertEqual(classify(path)[0], 'unresolved')

    def test_hash_shards_and_private_exclusion(self):
        h = hashlib.sha256(b'CHARLOTTE').hexdigest(); identity = 'hash-' + h
        self.assertEqual(classify(f'outputImages/{identity[:7]}/{identity[7:9]}/{identity}_300.webp')[0], 'text_path_candidate')
        self.assertEqual(classify('outputImages/GUIDaa/bb/GUIDabc_300.jpg')[0], 'private_or_mixed_unresolved')
        self.assertEqual(classify('outputMorphs/s1/s2/frame.jpg')[0], 'unresolved')

    def test_counts_coverage_and_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); name = 'CHARLOTTE'; key = 'hash-' + hashlib.sha256(name.encode()).hexdigest()
            path = root/f'outputImages/{key[:7]}/{key[7:9]}/{key}_300.jpg'
            path.parent.mkdir(parents=True); path.write_bytes(b'not an image: metadata only')
            (root/'unknown').write_bytes(b'other')
            report = scan([('fixture', str(root))], names=[name, 'OLIVER'])
            self.assertTrue(report['complete']); self.assertFalse(report['publicationApproved'])
            self.assertEqual(report['filesScanned'], 2)
            self.assertEqual(report['namesCoverage']['fixture'][name], ['300.jpg'])
            self.assertEqual(report['namesNotObserved'], ['OLIVER'])
            self.assertFalse(scan([('fixture', str(root))], max_files=1)['complete'])

    def test_missing_root_and_symlink(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root/'link').symlink_to(root, target_is_directory=True)
            self.assertFalse(scan([('fixture', str(root))])['complete'])
            self.assertFalse(scan([('missing', str(root/'absent'))])['complete'])


if __name__ == '__main__': unittest.main()
