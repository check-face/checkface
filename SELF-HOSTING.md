# Self-hosting the CheckFace API

This runs the same API that serves `api.facemorph.me`: `src/server/checkface.py` with NVIDIA's official
TensorFlow StyleGAN2 (`dnnlib` and its custom CUDA ops) and the official `stylegan2-ffhq-config-f` network.
It also runs the encoder4editing service behind `/api/encodeimage/`, and MongoDB for uploaded and registered
latents. The endpoints are documented in [docs/api.md](docs/api.md) and are unchanged.

Nothing here is a prebuilt image. You build locally from base images you pull yourself, and the build downloads
the model weights from their original publishers and checks their sha256 (see [Licensing](#licensing)).

## Prerequisites

- Linux x86_64 with an NVIDIA GPU, at least 8 GB VRAM for the default batch size (see [Memory](#memory-and-batch-size)).
- An NVIDIA driver for your GPU. The `legacy` base (CUDA 10.0) needs driver 410 or newer; the `ngc` base needs a
  driver supporting CUDA 12.1, i.e. 530 or newer. Newer drivers run older CUDA builds.
- Docker Engine with Compose v2 and BuildKit (the default since Docker 23).
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
  Check it works with `docker run --rm --gpus all ubuntu nvidia-smi`.
- About 30 GB of disk for the `ngc` build (12 GB for `legacy`), and network access to Docker Hub, `nvcr.io`
  (for `ngc`), NVIDIA's CDN, Google Drive and dlib.net while building.

## Pick the base for your GPU

The generator needs TensorFlow 1.15, which upstream only built for CUDA 10.0. NVIDIA kept TF1 alive in its NGC
containers until release 23.03, so there are two bases. Both run the identical StyleGAN2 code path.

| `CHECKFACE_TF_BASE` | Base image | CUDA / cuDNN | GPUs |
|---|---|---|---|
| `legacy` (default) | `tensorflow/tensorflow:1.15.5-gpu-py3` | 10.0 / 7.6 | Kepler to Turing: GTX 9xx, GTX 10xx (the deployed GTX 1080), RTX 20xx, Titan V, V100, T4 |
| `ngc` | `nvcr.io/nvidia/tensorflow:23.03-tf1-py3` | 12.1 / 8.8 | Maxwell to Hopper: adds RTX 30xx, RTX 40xx, A100, L4/L40, H100 |

Blackwell GPUs (RTX 50xx, B100/B200, sm_100/sm_120) are not supported by any TensorFlow 1 build, including the
NGC 23.03 container (its entrypoint checks SM 5.2–9.0), so the TensorFlow StyleGAN2 cannot run on them.

The encoder uses PyTorch, chosen with `CHECKFACE_ENCODER_TORCH`: `cu126` (default, sm_50–sm_90: GTX 9xx to H100),
`cu128` (sm_70–sm_120: Volta to Blackwell) or `cpu`. If the GPU isn't covered by the wheel, the encoder runs on CPU
automatically. That is fast enough for one 256 px encode per request.

## Build and run

```console
git clone https://github.com/check-face/checkface.git
cd checkface/src/server

# GTX 10xx / RTX 20xx and other pre-Ampere GPUs:
docker compose up -d --build

# RTX 30xx / 40xx, A100, H100:
CHECKFACE_TF_BASE=ngc docker compose up -d --build
```

The first build downloads about 3 GB of base images (about 16 GB for `ngc`) and:
- `stylegan2-ffhq-config-f.pkl` from `https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/` (381,673,535 bytes, sha256 `adf127ea…948d`);
- `e4e_ffhq_encode.pt` from the encoder4editing Google Drive link, and `shape_predictor_68_face_landmarks.dat` from dlib.net, both sha256-checked.

On first start the API compiles the two StyleGAN2 CUDA ops for your GPU with the image's `nvcc` (about 30 s),
warms up the generator, then logs `Generator ready`. On the `ngc` base, the first requests for each new batch size
are slow (up to about 30 s in total) while cuDNN autotunes; after that it runs at full speed.

Services and ports (unchanged from the deployment):
- API: `http://localhost:8080`
- Prometheus metrics: `:8000` (`generator_network_seconds`, `request_processing_seconds`, `ffmpeg_processing_seconds`, `image_generating`, `image_encoding`, `job_queue`)
- encoder: `:8083`
- MongoDB: internal only

Generated images and morphs are cached in `./checkfacedata`, or in `CHECKFACE_DATA_DIR` if set. On the original host
that is `CHECKFACE_DATA_DIR=/mnt/unit/checkfacedata`. The MongoDB volume is named `checkfacedbdata`: an existing
deployment keeps using it, and a fresh host creates it.

## Verify

```console
docker compose logs -f api          # wait for "Generator ready"
curl -o face.jpg 'http://localhost:8080/api/face/?seed=1&dim=1024'
curl -o morph.mp4 'http://localhost:8080/api/mp4/?from_seed=3&to_seed=4&num_frames=26&dim=1024'
curl -F usrimg=@../../docs/assets/images/face.jpg -F tryalign=true http://localhost:8080/api/encodeimage/
curl 'http://localhost:8080/api/face/?guid=<guid from the previous call>&dim=512' -o encoded.jpg
```

You can compare your output with the hosted API's
(`https://api.facemorph.me/api/face/?seed=1&dim=1024`). Both are JPEG encodes of the same generator output, so the
files differ by only a few levels in a fraction of a percent of pixels.

Without a GPU, `docker run --rm <api image> python ci_smoke.py sm_61` checks the checkpoint, compiles the CUDA ops and
serves the non-generating routes. `docker run --rm -v $PWD/face.jpg:/face.jpg <encoder image> python ci_smoke.py /face.jpg`
encodes on CPU. CI runs both (`.github/workflows/self-host-images.yml`).

## Memory and batch size

As deployed, the worker batches up to `GENERATOR_BATCH_SIZE` queued images (default 10) into one generator call.
On a dedicated 8 GB card (the deployed GTX 1080) that fits. If other processes share the GPU, or the card has less
memory, lower it, e.g. `GENERATOR_BATCH_SIZE=2 docker compose up -d`. `LOW_GPU_MEM=true` instead caps TensorFlow at
75% of GPU memory with unified-memory oversubscription, which is slower but avoids out-of-memory errors. The encoder
also takes GPU memory; `CHECKFACE_ENCODER_DEVICE=cpu` keeps it on the CPU and leaves the GPU to the generator.

If a generator call runs out of memory, the worker thread stops, as in the deployment, and requests time out after
30 s. Restart the `api` service with a smaller batch.

## Measured

Measured with `docker compose` on an RTX 2080 SUPER (8 GB, about 3 GB used by a desktop session), with
`GENERATOR_BATCH_SIZE=2` and `CHECKFACE_ENCODER_DEVICE=cpu`. Every route in docs/api.md was exercised, including
upload, encode, guid and morph.

| | legacy base | ngc base | hosted API (GTX 1080, batch 10) |
|---|---|---|---|
| Generator, steady state | 30–35 ms per 1024 px image | 30 ms per image | about 57 ms per image |
| Uncached `/api/face/?dim=1024` after the first | 53–55 ms | 55 ms once autotuned (first ones 0.2–2.9 s) | — |
| Uncached 26-frame 1024 px `/api/mp4/` | 1.35 s (first: 4.8 s) | 1.26–1.28 s (first: 6.8 s) | about 2.9 s |
| Difference vs the hosted API's JPEGs, 4 faces at dim 1024 | 0.02–0.26% of pixels, ≤6 levels, PSNR 70.6–83 dB | 0.07–0.33% of pixels, ≤8 levels, PSNR 69.8–76 dB | — |

The remaining differences are GPU/cuDNN rounding (GTX 1080 vs RTX 2080 SUPER) that occasionally flips a JPEG
coefficient. The encoder's W+ output on GPU and CPU agrees to within 1e-5.

## What was changed to make this reproducible

- `tensorflow/tensorflow:1.15.5-gpu-py3` ships NVIDIA CUDA apt repos whose signing key NVIDIA rotated in 2022, so the
  original `RUN apt update` fails (`NO_PUBKEY A4B469963BF863CC`). The Dockerfile removes those two repo entries,
  since only Ubuntu's `ffmpeg` is installed.
- The encoder image `olicoad/encoder4editing:latest` is no longer on Docker Hub. `src/encoder` rebuilds it from the
  deployed [check-face/encoder4editing](https://github.com/check-face/encoder4editing) server code with the same HTTP
  contract. It loads only the e4e encoder and `latent_avg`, because the deployed server never used the StyleGAN decoder it also built.
- On hosts where the NVIDIA Container Toolkit mounts the driver at `/usr/lib` (Arch, Fedora/RHEL), the CUDA 10.0
  image's libcuda *stub*, which it registers in `ld.so.conf`, shadowed the real driver and `cuInit` failed. The legacy
  stage drops that loader entry. Debian/Ubuntu hosts were not affected.
- Weights download from NVIDIA's current CDN, the URL linked from NVlabs/stylegan2, instead of the old CloudFront
  mirror, and the build fails if the file isn't the pinned original.
- The new `ngc` base adds Ampere, Ada and Hopper support. Compose no longer hard-codes the deployment host's data path
  or requires a pre-created external DB volume.

## Licensing

This repository ships source only: Dockerfiles, our code and docs. It ships no container images and no model
weights. You pull base images and download weights yourself, and the terms below apply to your use of them. The
models make any deployment research/evaluation (non-commercial) only.

| Component | Terms | What it means here |
|---|---|---|
| StyleGAN2 code in `src/server/dnnlib`, ops, `pretrained_networks.py`, and `stylegan2-ffhq-config-f.pkl` | [Nvidia Source Code License-NC](https://nvlabs.github.io/stylegan2/license.html) (also published next to the weights on NVIDIA's CDN) | "used or intended for use non-commercially … for research or evaluation purposes only". Redistributing the code requires keeping this license and its notices (see `src/server/LICENSE.txt`). |
| FFHQ (training data of the models) | [CC BY-NC-SA 4.0](https://github.com/NVlabs/ffhq-dataset) | Non-commercial. |
| `tensorflow/tensorflow:1.15.5-gpu-py3` | TensorFlow Apache-2.0. CUDA 10.0 and cuDNN 7.6 inside come from `nvidia/cuda` and are covered by the [CUDA EULA](https://docs.nvidia.com/cuda/eula/index.html) and the cuDNN license. | Pull it and build on it locally. We publish no derived image. |
| `nvcr.io/nvidia/tensorflow:23.03-tf1-py3` | The container states it is governed by the [NVIDIA Deep Learning Container License](https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license): "By pulling and using the container, you accept the terms". NGC currently also cites the [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/). | You accept these when you pull. Building FROM it locally for your own use is permitted. NVIDIA's proprietary parts run on NVIDIA GPUs only. We do not redistribute it or images derived from it, and you should not publish your built image either. |
| encoder4editing code (`src/encoder/e4e`) | [MIT](https://github.com/omertov/encoder4editing/blob/main/LICENSE), see `src/encoder/e4e/LICENSE` | — |
| `e4e_ffhq_encode.pt` | No terms stated. It is trained against the StyleGAN2 FFHQ generator and data. | Treat as non-commercial. |
| `shape_predictor_68_face_landmarks.dat` | dlib's model, trained on iBUG 300-W ([dlib-models](https://github.com/davisking/dlib-models)) | "the trained model therefore can't be used in a commercial product". |
| PyTorch wheels | BSD-3-Clause; the bundled CUDA libraries fall under the CUDA EULA | Installed from download.pytorch.org at build time. |
| `mongo:4.2` | SSPL v1 | Fine for unmodified use as the internal database. |
| NVIDIA Container Toolkit | Apache-2.0 | Host prerequisite. |

Your locally built images contain the downloaded weights and NVIDIA libraries, so don't push them to a public
registry. For commercial use of the StyleGAN2 models, contact NVIDIA Research licensing
(https://www.nvidia.com/en-us/research/inquiries/).
