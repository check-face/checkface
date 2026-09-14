"""Fast integrity and setup tests: mismatches must fail before pickle loading."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from encoder_assets import ASSETS, verified_asset
from prepare_encoder import download_asset

class AssetTests(unittest.TestCase):
    def test_existing_corrupt_asset_is_rejected_and_preserved_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); name='e4e_ffhq_encode.pt'; target=root/name
            target.write_bytes(b'corrupt model')
            with patch('urllib.request.urlopen') as network:
                with self.assertRaises(ValueError): download_asset(name,root)
                network.assert_not_called()
            self.assertEqual(target.read_bytes(),b'corrupt model')

    def test_same_size_wrong_hash_is_rejected_before_deserialization(self):
        # Small checked fixture exercises the real hash gate without a 1.2GB allocation.
        good=b'expected checkpoint'; bad=b'corrupt! checkpoint'
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); name='e4e_ffhq_encode.pt'
            (root/name).write_bytes(bad)
            entry={**ASSETS[name],'bytes':len(bad),'sha256':hashlib.sha256(good).hexdigest()}
            with patch.dict(ASSETS,{name:entry}):
                with self.assertRaisesRegex(ValueError,'checksum'): verified_asset(name,root)

    def test_encoder_checks_asset_before_torch_load(self):
        import encoder
        with patch('encoder.verified_asset',side_effect=ValueError('checksum mismatch')), patch('encoder.torch.load') as loader:
            with self.assertRaises(ValueError): encoder.Encoder()
            loader.assert_not_called()

if __name__=='__main__': unittest.main()
