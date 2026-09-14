"""Deployed e4e preprocessing/encoder on CPU; W+ goes directly to synthesis.

API owns the inference_lock shared with synthesis. This module additionally guards
its lazy singleton for standalone callers. Never loads uploaded model checkpoints.
"""
import io
import threading
from types import SimpleNamespace
import warnings

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch

from encoder_assets import verified_asset, verify_assets
from e4e.models.encoders.psp_encoders import Encoder4Editing

Image.MAX_IMAGE_PIXELS = 8192 * 4096  # Same deployed encoder limit.
_lock = threading.RLock()
_encoder = None

class Encoder:
    def __init__(self):
        # Digest and byte count must match before trusted checkpoint deserialization.
        checkpoint = torch.load(verified_asset('e4e_ffhq_encode.pt'), map_location='cpu', weights_only=False)
        opts = SimpleNamespace(**checkpoint['opts'])
        if opts.encoder_type != 'Encoder4Editing' or opts.stylegan_size != 1024:
            raise ValueError('Unexpected deployed encoder architecture')
        self.net = Encoder4Editing(50, 'ir_se', opts)
        state = {k.removeprefix('encoder.'): v for k, v in checkpoint['state_dict'].items() if k.startswith('encoder.')}
        self.net.load_state_dict(state, strict=True)
        self.net.eval().requires_grad_(False).to('cpu')
        self.latent_avg = checkpoint['latent_avg'].to('cpu') if opts.start_from_latent_avg else None
        self.predictor = None

    @torch.inference_mode()
    def encode(self, image, try_align):
        did_align = False
        if try_align:
            import dlib
            from e4e.utils.alignment import align_face, NumberOfFacesError
            if self.predictor is None:
                self.predictor = dlib.shape_predictor(str(verified_asset('shape_predictor_68_face_landmarks.dat')))
            try:
                image = align_face(None, self.predictor, img=image)
                did_align = True
            except NumberOfFacesError:
                pass  # Deployed no-face behavior: encode the original unaligned image.
        image = image.resize((256, 256), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        batch = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)
        codes = self.net((batch - 0.5) / 0.5)
        if self.latent_avg is not None:
            codes = codes + self.latent_avg.unsqueeze(0)
        if codes.shape != (1, 18, 512) or not torch.isfinite(codes).all():
            raise RuntimeError('Invalid encoder output')
        return codes[0].cpu().numpy().copy(), did_align

def warmup():
    global _encoder
    with _lock:
        if _encoder is None:
            _encoder = Encoder()

def encode_image(image_bytes: bytes, try_align: bool) -> tuple[np.ndarray, bool]:
    if not isinstance(image_bytes, bytes) or not image_bytes or type(try_align) is not bool:
        raise ValueError('An uploaded image and boolean alignment option are required')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(image_bytes)) as opened:
                image = opened.convert('RGB')
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ValueError('Invalid or oversized uploaded image') from error
    with _lock:
        warmup()
        return _encoder.encode(image, try_align)
