# encoder4editing service for /api/encodeimage/ (replaces the olicoad/encoder4editing image, no longer published).
#
# TORCH_CUDA picks the PyTorch 2.9.1 wheel build (see SELF-HOSTING.md):
#   cu126  Maxwell..Hopper (sm_50-sm_90; GTX 9xx/10xx, RTX 20xx/30xx/40xx, A100, H100)   [default]
#   cu128  Volta..Blackwell (sm_70-sm_120; adds RTX 50xx, B200; drops GTX 10xx)
#   cpu    no GPU; a single 256px encode still takes well under a second on a modern CPU
# With ENCODER_DEVICE=auto (default) the server falls back to CPU if the GPU is not covered by the wheel.
ARG TORCH_CUDA=cu126

FROM python:3.11-slim-bookworm AS download-models
COPY fetch_models.py /fetch_models.py
RUN python /fetch_models.py /models e4e_ffhq_encode.pt
RUN python /fetch_models.py /models shape_predictor_68_face_landmarks.dat

FROM python:3.11-slim-bookworm AS dlib-wheel
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake && rm -rf /var/lib/apt/lists/*
RUN pip wheel --no-cache-dir --no-deps -w /wheels dlib==20.0.1

FROM python:3.11-slim-bookworm
ARG TORCH_CUDA
RUN pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/${TORCH_CUDA}
COPY --from=dlib-wheel /wheels /wheels
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --find-links /wheels -r /app/requirements.txt && rm -rf /wheels
WORKDIR /app
COPY . /app
COPY --from=download-models /models /app/pretrained_models
EXPOSE 8080
CMD ["python", "server_inference.py"]
