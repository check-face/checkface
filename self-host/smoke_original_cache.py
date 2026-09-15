"""Real HTTP synthesis deduplication and restart reuse; run against own candidate.

python smoke_original_cache.py --state /tmp/original-smoke.json
# restart API (preserve volumes), then:
python smoke_original_cache.py --state /tmp/original-smoke.json --after-restart
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import secrets
from urllib.request import urlopen
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='http://127.0.0.1:8080')
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--after-restart', action='store_true')
    args = parser.parse_args()
    def get(path):
        with urlopen(args.base.rstrip('/') + path, timeout=600) as response:
            return response.read()
    def health():
        return json.loads(get('/healthz'))['original_cache']
    seed = json.loads(args.state.read_text())['seed'] if args.after_restart else secrets.randbelow(2**32)
    before = health()
    dimensions = [513, 777] if args.after_restart else [127, 256, 1024]
    def face(dim):
        file_format = 'webp' if dim % 2 else 'jpg'
        data = get(f'/api/face/?seed={seed}&dim={dim}&format={file_format}')
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            assert image.size == (dim, dim), (dim, image.size)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(face, dimensions))
    after = health()
    expected = 0 if args.after_restart else 1
    assert after['generated'] - before['generated'] == expected, (before, after)
    assert after['write_failures'] == before['write_failures'], (before, after)
    assert after['hits'] - before['hits'] >= len(dimensions) - expected, (before, after)
    if not args.after_restart:
        args.state.write_text(json.dumps({'seed': seed}) + '\n')
    print(json.dumps({'after_restart': args.after_restart, 'before': before, 'after': after}))


if __name__ == '__main__':
    main()
