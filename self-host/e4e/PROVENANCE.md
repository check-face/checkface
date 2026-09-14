# Deployed e4e encoder source

This minimal source subset is copied from the previously verified CPU adapter in
`facemorph.me/experiment/hf/vendor/e4e`, itself copied read-only from the deployed
`olicoad/encoder4editing` container on 11 September 2026. Upstream is
[omertov/encoder4editing](https://github.com/omertov/encoder4editing); its MIT license
is retained in `LICENSE`. The directory name of the older experiment is provenance,
not a dependency on hosted inference or authentication.

Reuse includes the original encoder, helpers and face alignment. Existing CPU
compatibility changes preserve relative imports, extract the encoder's ordinary
`EqualLinear` branch from the decoder (avoiding CUDA extension imports), and replace
Pillow's removed ANTIALIAS alias with Resampling.LANCZOS. No decoder, Gradio server,
training code or model weights are bundled here. Decoder synthesis uses the separate
verified StyleGAN2 adapter.

`encoder.py` retains deployed RGB conversion, largest-face alignment, no-face
fallback, bilinear 256px resize, normalization, strict checkpoint state and latent
average addition. Resulting W+ goes directly to synthesis, without mapping/truncation.
Modern runtime libraries are not a claim of bit-identical production execution.

`prepare_encoder.py` downloads upstream model files only after explicit permitted-use
acknowledgement and checks exact deployed byte counts/SHA-256. The official e4e FFHQ
Google Drive endpoint returned the expected 1,201,649,680-byte attachment on
14 September 2026. The dlib landmark model is trained on iBUG 300-W; upstream warns
that its dataset excludes commercial use. Code's MIT license does not override
model/data terms. Neither models nor container images are published by this work.
