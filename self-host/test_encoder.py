"""Real deployed-weight CPU encoder regression; no synthetic stub inference."""
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

import encoder

def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    fixtures = Path(__file__).with_name('fixtures')
    encoder.verify_assets()
    results = []
    for seed in (0, 1):
        image = (fixtures/f'seed-{seed}.png').read_bytes()
        latent, aligned = encoder.encode_image(image, True)
        repeated, repeated_align = encoder.encode_image(image, True)
        assert aligned is True and repeated_align is True
        assert latent.shape == (18,512) and latent.dtype == np.float32 and np.isfinite(latent).all()
        assert np.array_equal(latent,repeated), 'Encoding is not repeatable'
        reference = np.load(fixtures/f'seed-{seed}-encoded-w-plus.npy',allow_pickle=False)
        error=float(np.max(np.abs(latent-reference)))
        assert error <= 0.0001, f'Deployed CPU adapter reference changed: {error}'
        _, unaligned = encoder.encode_image(image, False)
        assert unaligned is False
        results.append({'seed':seed,'did_align':aligned,'repeat_exact':True,'max_reference_error':error})
    buffer=io.BytesIO(); Image.new('RGB',(256,256),(128,128,128)).save(buffer,format='PNG')
    fallback, did_align=encoder.encode_image(buffer.getvalue(),True)
    direct, direct_align=encoder.encode_image(buffer.getvalue(),False)
    assert did_align is False and direct_align is False and np.array_equal(fallback,direct)
    for image, align in [(b'not an image',True),(b'',True),(buffer.getvalue(),'true')]:
        try: encoder.encode_image(image,align)
        except ValueError: pass
        else: raise AssertionError('Invalid input was accepted')
    print(json.dumps({'encoder_cases':results,'no_face_fallback_exact':True,'malformed_uploads_rejected':True,'device':'cpu','torch':torch.__version__}))

if __name__ == '__main__': main()
