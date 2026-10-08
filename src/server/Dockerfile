# CheckFace API: NVIDIA's official TensorFlow StyleGAN2 (dnnlib + custom CUDA ops), as deployed.
#
# Pick the TensorFlow 1.15 base for your GPU generation (see SELF-HOSTING.md):
#   TF_BASE=legacy  tensorflow/tensorflow:1.15.5-gpu-py3   CUDA 10.0  Kepler..Turing (GTX 10xx, RTX 20xx; the deployed setup)
#   TF_BASE=ngc     nvcr.io/nvidia/tensorflow:23.03-tf1-py3 CUDA 12.1  Pascal..Hopper (adds RTX 30xx/40xx, A100, H100)
# Base images are pulled by you from their publishers; nothing derived from them is published by this project.
ARG TF_BASE=legacy

FROM python:3.7-slim AS download-model
RUN pip install --no-cache-dir numpy==1.21.6 requests==2.31.0
COPY ./dnnlib /app/dnnlib
COPY ./pretrained_networks.py ./fetchModel.py /app/
WORKDIR /app
# Downloads stylegan2-ffhq-config-f.pkl from NVIDIA and checks its pinned size and sha256.
RUN python fetchModel.py

FROM tensorflow/tensorflow:1.15.5-gpu-py3 AS base-legacy
# NVIDIA rotated its apt signing key in 2022, so the CUDA apt repos baked into this image make `apt update` fail.
# Nothing is installed from them (only Ubuntu's ffmpeg), so drop them.
RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list
# The image puts CUDA's libcuda *stub* on the loader path (so TensorFlow imports without a GPU). Where the NVIDIA
# Container Toolkit mounts the host driver at /usr/lib or /usr/lib64 (Arch, Fedora, RHEL hosts) rather than
# /usr/lib/x86_64-linux-gnu (Debian/Ubuntu hosts), the stub shadows the real driver and cuInit fails. Drop it.
RUN rm -f /etc/ld.so.conf.d/z-cuda-stubs.conf && ldconfig
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

FROM nvcr.io/nvidia/tensorflow:23.03-tf1-py3 AS base-ngc
# Keep NVIDIA's TensorFlow/numpy; add only the API's own dependencies (same pins as requirements.txt).
COPY ./requirements-ngc.txt /app/requirements-ngc.txt
RUN pip install --no-cache-dir -r /app/requirements-ngc.txt

FROM base-${TF_BASE}
RUN apt-get update && apt-get install --no-install-recommends -y ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=download-model /app/.stylegan2-cache /app/.stylegan2-cache
COPY . /app
CMD ["python", "checkface.py"]
