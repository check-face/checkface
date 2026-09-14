"""Download the official fixed checkpoint; never download or deserialize arbitrary models."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

SOURCE_URL = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-ffhq-config-f.pkl'
SOURCE_SHA256 = 'adf127ea7bb8a7788c8bdeda3c9937f7310b669b09ecf799ca53a631ff46948d'
SOURCE_BYTES = 381673535
MODEL_NAME = 'stylegan2-ffhq-config-f.pkl'
STYLEGAN_REV = 'd72cc7d041b42ec8e806021a205ed9349f87c6a4'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accept-research-license', action='store_true', help='Confirm you reviewed NVIDIA research/evaluation-only terms and your use is permitted')
    parser.add_argument('--directory', type=Path, default=Path('/models'))
    args = parser.parse_args()
    if not args.accept_research_license:
        parser.error('Read README licensing links first; permitted research/evaluation setup requires --accept-research-license')
    args.directory.mkdir(parents=True, exist_ok=True)
    target = args.directory / MODEL_NAME
    if target.exists():
        if digest(target) != SOURCE_SHA256:
            raise SystemExit('Existing model checksum mismatch; move it aside and retry. It was not loaded or overwritten.')
    else:
        temporary = target.with_suffix('.download')
        try:
            sha, size = hashlib.sha256(), 0
            with urllib.request.urlopen(SOURCE_URL, timeout=120) as source, temporary.open('wb') as output:
                while chunk := source.read(1024 * 1024):
                    size += len(chunk)
                    if size > SOURCE_BYTES:
                        raise RuntimeError('Download exceeds pinned model size')
                    sha.update(chunk)
                    output.write(chunk)
            if size != SOURCE_BYTES or sha.hexdigest() != SOURCE_SHA256:
                raise RuntimeError('Official download differs from pinned source; no model installed')
            temporary.chmod(0o644)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    (args.directory / 'provenance.json').write_text(json.dumps({
        'source_url': SOURCE_URL, 'source_sha256': SOURCE_SHA256, 'source_bytes': SOURCE_BYTES,
        'converter': 'NVlabs/stylegan2-ada-pytorch legacy.py', 'converter_revision': STYLEGAN_REV,
        'conversion': 'Verified original checkpoint converted in memory at API startup; no TensorFlow or retraining',
        'terms': 'https://github.com/NVlabs/stylegan2/blob/master/LICENSE.txt',
    }, indent=2) + '\n')
    print(f'Verified {target}: {SOURCE_SHA256}')


if __name__ == '__main__':
    main()
