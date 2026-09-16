"""Private, lossless originals in a namespace separate from historic derivatives."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
import threading
from PIL import Image, PngImagePlugin, UnidentifiedImageError


class OriginalCache:
    def __init__(self, root, max_bytes=10 * 1024**3):
        self.root = Path(root) / 'originals-v1'
        self.max_bytes = int(max_bytes)
        if self.max_bytes < 0:
            raise ValueError('CHECKFACE_ORIGINAL_CACHE_BYTES must be non-negative')
        self._lock = threading.RLock()
        self.hits = self.generated = self.write_failures = 0

    @contextmanager
    def locked(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with self._lock, (self.root / '.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                # A crashed writer may leave staging, never a valid cache hit.
                for path in self.root.glob('.pending-*'):
                    path.unlink(missing_ok=True)
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def get_or_create(self, identity, generate):
        specification = json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False)
        key = hashlib.sha256(specification.encode()).hexdigest()
        # Lock acquisition/storage denial does not prevent generation. Once inside
        # the lock, inference errors must propagate, never trigger a second attempt.
        with self._lock:
            try:
                lock = self.locked()
                lock.__enter__()
            except OSError:
                self.write_failures += 1
                logging.getLogger(__name__).warning('Original cache unavailable; result will not persist')
                self.generated += 1
                return generate()
            try:
                path = self.root / (key + '.png')
                try:
                    with Image.open(path) as stored:
                        stored.load()
                        expected = stored.info.get('pixels_sha256')
                        if (stored.size == (1024, 1024) and stored.mode == 'RGB'
                                and stored.info.get('generation') == specification
                                and hashlib.sha256(stored.tobytes()).hexdigest() == expected):
                            self.hits += 1
                            return stored.copy()
                except (OSError, ValueError, UnidentifiedImageError):
                    pass
                result = generate()
                self.generated += 1
                if result.size != (1024, 1024) or result.mode != 'RGB':
                    raise ValueError('Original must be raw 1024x1024 RGB pixels')
                staging = None
                try:
                    info = PngImagePlugin.PngInfo()
                    info.add_text('generation', specification)
                    info.add_text('pixels_sha256', hashlib.sha256(result.tobytes()).hexdigest())
                    with tempfile.NamedTemporaryFile(dir=self.root, prefix='.pending-', delete=False) as stream:
                        staging = Path(stream.name)
                        result.save(stream, format='PNG', pnginfo=info)
                        stream.flush()
                        os.fsync(stream.fileno())
                    used = sum(p.stat().st_size for p in self.root.glob('*.png') if p != path)
                    if used + staging.stat().st_size > self.max_bytes:
                        raise OSError('Original cache quota reached')
                    os.replace(staging, path)
                    directory = os.open(self.root, os.O_RDONLY)
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
                except OSError:
                    self.write_failures += 1
                    logging.getLogger(__name__).warning('Original could not be persisted (quota or storage failure)')
                finally:
                    if staging is not None:
                        try:
                            staging.unlink(missing_ok=True)
                        except OSError:
                            pass  # A later successful lock acquisition cleans staging.
                return result
            finally:
                lock.__exit__(None, None, None)
