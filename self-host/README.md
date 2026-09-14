# CheckFace self-host CPU API

A local **research/evaluation candidate** with modern Python/PyTorch inference.
Build it yourself from the candidate branch; no prebuilt image or release is
published. It runs without an NVIDIA container, GPU driver, HF service, Mongo or
Triton. It supports deterministic seeds and words, and short linear morphs.
It is not a drop-in replacement for the legacy public API.

## Fresh-clone setup

Install Docker Engine/Desktop with Compose v2. Allow several GiB RAM and disk
space for Python/PyTorch and the 364 MiB official model. Read the licensing
section below: this setup is for permitted research/evaluation use.

```sh
git clone --branch candidate/self-host-api https://github.com/check-face/checkface.git
cd checkface/self-host
docker compose build
docker compose run --rm setup --accept-research-license
docker compose up -d --wait --wait-timeout 300 api
docker compose exec -T api python smoke.py
```

The setup command downloads the original
[official NVIDIA FFHQ config-F checkpoint](https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-ffhq-config-f.pkl)
to a Docker volume on your machine. It verifies size and the pinned SHA-256
**before any deserialization**. No weights enter the image or this repository.
Interrupted or incorrect downloads are discarded; an existing incorrect model
is rejected rather than replaced. Rerunning setup verifies and reuses a correct
model. Download provenance is saved in the volume.

At API startup, NVIDIA's pinned `legacy.py` converts the verified TensorFlow
checkpoint into PyTorch in memory. TensorFlow is not installed or run. This
conversion repeats at startup; it requires no training, account or secret.
The API serves only after model load/conversion succeeds. CPU inference can take
seconds per face; the smoke generates several real 1024px faces and can take
minutes on a small CPU. Health polling is delayed while a request is rendering.

```sh
curl 'http://127.0.0.1:8080/healthz'
curl 'http://127.0.0.1:8080/v1/face?seed=0' --output face.png
curl 'http://127.0.0.1:8080/v1/face?text=hello' --output hello.png
curl 'http://127.0.0.1:8080/v1/morph?from=0&to=1&frames=3' --output morph.zip
docker compose logs api
docker compose down
```

`down` retains the model volume for reuse. `down --volumes` deletes this local
Compose project's model volume, so setup must download it again.

## Supported contract

| Route | Behavior |
|---|---|
| `/healthz` | Loaded CPU provider, API version and source model hash. |
| `/v1/face?seed=0` | 1024px RGB PNG. Seed is a uint32 integer. |
| `/v1/face?text=hello` | 1024px RGB PNG. SHA-256 UTF-8 digest interpreted as little-endian uint32 values seeds NumPy RandomState, matching the current legacy text latent rule. Text is at most 512 UTF-8 bytes; use URL encoding. |
| `/v1/morph?from=0&to=1&frames=3` | ZIP of 2–12 ordered PNG frames and reproducible JSON manifest. Both endpoints are included; interpolation is linear Z. |

Constant noise, truncation psi 0.7/cutoff 8, FP32 synthesis and the same config-F
model are used throughout. This linear morph differs from classic sinusoidal
looping. Inputs reject ambiguous seed+text, duplicate parameters, out-of-range
seeds and excessive frame counts. Errors use HTTP status codes. Unknown legacy
routes return 404; they are not silently redirected to a webpage.

The service publishes localhost only, runs non-root with a read-only filesystem
and model volume, and serializes inference to bound working memory. This small
HTTP server is intended for local evaluation; public hosting requires a proper
application server, authentication, request timeouts and resource controls.

## Validation and reproducibility

`smoke.py` calls the running API and **requires real model inference**. It checks:

- Repeatable 1024px output, different seeds and deterministic word identity.
- Every pixel against independent converted CPU seed-0/seed-1 references after
  the established 512px resize, with maximum RGB error 1. See `fixtures/README.md`.
- Morph endpoint equality, a distinct midpoint, and manifest contents.
- Invalid request bounds, unsupported legacy routes, wrong-checkpoint startup
  rejection, and valid model provenance JSON when present.

The `Self-host CPU inference` GitHub workflow builds on Linux amd64, downloads
and verifies the official checkpoint, starts the API, and runs these checks on
candidate branch pushes/PRs or manual dispatch. Missing models or failed
conversion fail CI; there is no fake-inference or skipped-model pass. The job
publishes no image, model, installer or release artifact.

The source hash is
`adf127ea7bb8a7788c8bdeda3c9937f7310b669b09ecf799ca53a631ff46948d`,
size 381,673,535 bytes. NVIDIA converter revision:
`d72cc7d041b42ec8e806021a205ed9349f87c6a4`.
The local migration workspace's already-converted generator is additionally
accepted only at its fixed digest
`dc315131a0d8466167f69962dd75567db8815261a5c5cab545fe9b48fe815994`;
users of this branch do not need that private workspace artifact.

Python, torch and direct dependency versions and NVIDIA revision are pinned.
Full transitive hashes, base-image digest and distribution SBOM remain release
work. Fixed CPU reference tests are regression evidence for these workflows;
they do not prove historic JPEG byte parity or all 31 performance research cases.
The obsolete legacy image's availability is not needed by this build.

## Licensing and scope

Reviewed primary terms on 14 September 2026:

- [StyleGAN2 ADA license at the pinned revision](https://raw.githubusercontent.com/NVlabs/stylegan2-ada-pytorch/d72cc7d041b42ec8e806021a205ed9349f87c6a4/LICENSE.txt): redistribution requires retained terms/notices; use is restricted to research/evaluation. Its license remains at `/opt/stylegan/LICENSE.txt` in the image.
- [Original StyleGAN2 license](https://raw.githubusercontent.com/NVlabs/stylegan2/master/LICENSE.txt): also imposes research/evaluation restrictions. The [official repository](https://github.com/NVlabs/stylegan2) links the checkpoint source used here and [NVIDIA research licensing inquiries](https://www.nvidia.com/en-us/research/inquiries/).
- CheckFace's repository license is [CC BY-NC 4.0](../LICENSE.txt). Model rights,
  original checkpoint terms, base OS, Python/PyTorch and dependency notices need
  separate review before any distribution or broader deployment.

These limits affect **use**, not only publication. A local build is not an
exemption and a green build does not establish permission for general public or
commercial operation. `--accept-research-license` records the user's explicit
setup choice; it does not grant additional rights. No container or model is
published by this candidate or its CI.

Not implemented: uploaded-photo/e4e encoding, hosted GUID/archive lookup, full
legacy route compatibility, MP4 encoding, GPU runtime, or figure-eight/custom
morphs. These can evolve behind a versioned contract without changing this local
setup. CPU-only operation and existing classic services remain independent.
No live services, production mounts or other Triton infrastructure are changed.
