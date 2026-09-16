"""CPU evaluation API v1; deliberately separate from legacy public API routes."""
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
from urllib.parse import parse_qs, urlsplit
import zipfile
from prepare_model import SOURCE_SHA256, MODEL_NAME

MODEL_SHA256 = 'dc315131a0d8466167f69962dd75567db8815261a5c5cab545fe9b48fe815994'


def load_model():
    import torch
    from torch_utils.ops import bias_act, upfirdn2d, conv2d_gradfix
    from cpu_threads import configure_torch
    configure_torch(torch)
    torch.use_deterministic_algorithms(True)
    original_bias, original_up = bias_act.bias_act, upfirdn2d.upfirdn2d
    def bias(*args, **kwargs):
        kwargs['impl'] = 'ref'
        return original_bias(*args, **kwargs)
    def up(*args, **kwargs):
        kwargs['impl'] = 'ref'
        return original_up(*args, **kwargs)
    bias_act.bias_act, upfirdn2d.upfirdn2d = bias, up
    conv2d_gradfix.enabled = False
    data = Path(os.environ.get('CHECKFACE_MODEL', '/models/' + MODEL_NAME)).read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual == SOURCE_SHA256:
        import legacy
        # NVIDIA's pinned converter handles the original TensorFlow serialization
        # using stubs; TensorFlow itself is neither installed nor executed.
        model = legacy.load_network_pkl(io.BytesIO(data))['G_ema']
    elif actual == MODEL_SHA256:
        model = pickle.loads(data)  # Existing trusted converted workspace artifact.
    else:
        raise RuntimeError('Untrusted or wrong model: expected pinned CheckFace checkpoint')
    model._checkface_source_sha256 = actual
    return model.eval().requires_grad_(False).to('cpu')


def seed(query, key):
    value = query.get(key, [''])[0]
    if not value.isascii() or not value.isdecimal() or len(value) > 10:
        raise ValueError('Seeds must be integers in [0, 4294967295]')
    value = int(value)
    if value > 2**32 - 1:
        raise ValueError('Seed exceeds uint32')
    return value


def input_latent(query):
    import numpy as np
    if 'text' in query:
        if 'seed' in query:
            raise ValueError('Choose seed or text, not both')
        text = query['text'][0]
        if len(text.encode('utf-8')) > 512:
            raise ValueError('Text is limited to 512 UTF-8 bytes')
        identity = np.frombuffer(hashlib.sha256(text.encode('utf-8')).digest(), dtype='<u4')
        return np.random.RandomState(identity).randn(512)
    return np.random.RandomState(seed(query, 'seed')).randn(512)


def render(model, z):
    import numpy as np
    import torch
    from PIL import Image
    with torch.inference_mode():
        w = model.mapping(torch.from_numpy(np.asarray(z, dtype=np.float32)[None]), None,
                          truncation_psi=0.7, truncation_cutoff=8)
        out = model.synthesis(w, noise_mode='const', force_fp32=True, fused_modconv=False)
        if not torch.isfinite(out).all():
            raise RuntimeError('Non-finite synthesis output')
        pixels = (out.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0].numpy()
    output = io.BytesIO()
    Image.fromarray(pixels).save(output, format='PNG')
    return output.getvalue()


class BoundedHTTPServer(ThreadingHTTPServer):
    """Keep at most eight live handlers, including slow header/body clients."""
    daemon_threads = True

    def __init__(self, *args, **kwargs):
        self.slots = threading.BoundedSemaphore(8)
        super().__init__(*args, **kwargs)

    def process_request(self, request, client_address):
        request.settimeout(15)
        if not self.slots.acquire(blocking=False):
            try:
                request.settimeout(1)
                request.sendall(b'HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nRetry-After: 1\r\nConnection: close\r\n\r\n')
            except (OSError, TimeoutError):
                pass
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


def handler(model):
    inference = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def handle(self):
            try:
                super().handle()
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                # A disconnected/timed-out client is not a server failure.
                self.close_connection = True

        def do_GET(self):
            generating = urlsplit(self.path).path in ('/v1/face', '/v1/morph')
            if generating and not inference.acquire(blocking=False):
                self.send_response(503)
                self.send_header('Retry-After', '1')
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
            try:
                self.serve_get()
            finally:
                if generating:
                    inference.release()

        def serve_get(self):
            url = urlsplit(self.path)
            query = parse_qs(url.query, keep_blank_values=True)
            if len(self.path) > 4096 or any(len(v) != 1 for v in query.values()):
                self.send_error(400, 'Oversized or duplicate query parameters')
                return
            try:
                if url.path == '/healthz':
                    body, mime = json.dumps({'ready': True, 'provider': 'cpu', 'model_source_sha256': SOURCE_SHA256, 'api_version': 1, 'inference_busy': inference.locked()}).encode(), 'application/json'
                elif url.path == '/v1/face':
                    body, mime = render(model, input_latent(query)), 'image/png'
                elif url.path == '/v1/morph':
                    import numpy as np
                    a, b = [np.random.RandomState(seed(query, k)).randn(512) for k in ('from', 'to')]
                    frames = int(query.get('frames', ['3'])[0])
                    if not 2 <= frames <= 12:
                        raise ValueError('Use 2 to 12 frames')
                    output = io.BytesIO()
                    # Linear non-loop path includes both endpoints, unlike classic sinusoidal loops.
                    with zipfile.ZipFile(output, 'w') as archive:
                        for index, t in enumerate(np.linspace(0, 1, frames)):
                            archive.writestr(f'{index:03d}.png', render(model, a*(1-t) + b*t))
                        archive.writestr('manifest.json', json.dumps({'version': 1, 'path': 'linear-z', 'frames': frames,
                            'from': seed(query, 'from'), 'to': seed(query, 'to'), 'model_source_sha256': SOURCE_SHA256,
                            'truncation_psi': 0.7, 'truncation_cutoff': 8, 'noise': 'const', 'dimension': 1024}))
                    body, mime = output.getvalue(), 'application/zip'
                else:
                    self.send_error(404, 'Not supported by evaluation API v1')
                    return
            except (ValueError, OverflowError):
                self.send_error(400, 'Invalid seeds or frame count')
                return
            except Exception:
                self.send_error(500, 'Inference failed')
                return
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *_args):
            pass  # Do not log user inputs.
    return Handler


if __name__ == '__main__':
    # Health can proceed concurrently, while only one generation is admitted.
    BoundedHTTPServer(('0.0.0.0', 8080), handler(load_model())).serve_forever()
