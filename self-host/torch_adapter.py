"""The small TensorFlow-style inference surface used by the preserved API logic."""
from types import SimpleNamespace
import numpy as np
import torch


class TorchGenerator:
    def __init__(self, model):
        self.model = model
        self.input_shape = [None, 512]
        self.components = SimpleNamespace(mapping=SimpleNamespace(run=self.mapping),
                                          synthesis=SimpleNamespace(run=self.synthesis))

    def get_var(self, name):
        if name != 'dlatent_avg':
            raise KeyError(name)
        return self.model.mapping.w_avg.detach().cpu().numpy()

    @torch.inference_mode()
    def mapping(self, latents, _labels=None):
        z = torch.as_tensor(np.asarray(latents, dtype=np.float32), device='cpu')
        return self.model.mapping(z, None, truncation_psi=1).cpu().numpy()

    @torch.inference_mode()
    def synthesis(self, dlatents, **_legacy_options):
        images = []
        # Preserve full-resolution computation; limit runtime working memory to one face.
        for latent in np.asarray(dlatents, dtype=np.float32):
            ws = torch.from_numpy(latent[None]).to('cpu')
            out = self.model.synthesis(ws, noise_mode='const', force_fp32=True, fused_modconv=False)
            if not torch.isfinite(out).all():
                raise ValueError('Non-finite synthesis output')
            pixels = (out.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            images.append(pixels[0].cpu().numpy())
        return np.stack(images)

    def run(self, latents, _labels=None, **_legacy_options):
        ws = self.mapping(latents)
        average = self.get_var('dlatent_avg')
        ws[:, :8] = (ws[:, :8] - average) * 0.7 + average
        return self.synthesis(ws)
