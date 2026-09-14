"""Acquire original public encoder assets only after license acknowledgement."""
import argparse
import bz2
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request

from encoder_assets import ASSETS, verified_asset

def download_asset(name, directory):
    entry = ASSETS[name]
    target = directory / name
    if target.exists():
        verified_asset(name, directory)  # Refuse an existing mismatch; never overwrite it.
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=name+'.', suffix='.download', dir=directory, delete=False) as output:
            temporary = Path(output.name)
            digest = hashlib.sha256()
            total = 0
            with urllib.request.urlopen(entry['url'], timeout=120) as response:
                source = bz2.BZ2File(response) if entry.get('compression') == 'bz2' else response
                while chunk := source.read(1024 * 1024):
                    total += len(chunk)
                    if total > entry['bytes']:
                        raise ValueError(f'Official download exceeds expected size: {name}')
                    digest.update(chunk)
                    output.write(chunk)
            if total != entry['bytes'] or digest.hexdigest() != entry['sha256']:
                raise ValueError(f'Official download did not match deployed checksum: {name}')
        temporary.chmod(0o644)
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accept-research-license', action='store_true', help='Acknowledge permitted use under e4e/FFHQ and dlib iBUG model terms; no general commercial-use grant')
    parser.add_argument('--directory', type=Path, default=Path('/models'))
    args = parser.parse_args()
    if not args.accept_research_license:
        parser.error('Read the e4e/FFHQ and dlib/iBUG model terms, then pass --accept-research-license for permitted use')
    args.directory.mkdir(parents=True, exist_ok=True)
    for name in ASSETS:
        download_asset(name, args.directory)
        print(f'Verified {name}: {ASSETS[name]["sha256"]}', flush=True)
    (args.directory/'encoder-provenance.json').write_text(json.dumps({
        'assets': ASSETS,
        'acknowledgement': 'Operator acknowledged model terms for permitted research use; no redistribution or commercial grant implied.',
        'license_sources': ['https://github.com/omertov/encoder4editing/blob/main/LICENSE', 'https://github.com/NVlabs/ffhq-dataset#licenses', 'https://dlib.net/face_landmark_detection.py.html'],
    }, indent=2)+'\n')

if __name__ == '__main__':
    main()
