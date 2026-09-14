FROM --platform=$BUILDPLATFORM node:22-bookworm-slim AS node-tools
FROM --platform=$BUILDPLATFORM mcr.microsoft.com/dotnet/sdk:5.0.408-bullseye-slim@sha256:40c6bd0059eaa06b4a9c91cd3e6df138f6224bd02b2882bf6ce3aa4af3835fc5 AS frontend
COPY --from=node-tools /usr/local/ /usr/local/
ENV DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1
ARG FRONTEND_REV=0abb215f27b16e17e3919cf78b616b6ae4998a5d
RUN git clone https://github.com/check-face/facemorph.me.git /src && cd /src && git checkout "$FRONTEND_REV" && rm -rf .git
WORKDIR /src
COPY frontend-overlay/source/ ./
RUN npm ci && npm run build:self-host

FROM python:3.11-slim-bookworm AS runtime-deps
ARG STYLEGAN_REV=d72cc7d041b42ec8e806021a205ed9349f87c6a4
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates ffmpeg build-essential cmake && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git /opt/stylegan && cd /opt/stylegan && git checkout "$STYLEGAN_REV" && rm -rf .git
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements.txt

FROM runtime-deps AS api-base
COPY *.py ./
COPY templates ./templates
COPY assets ./checkfacedata/assets
COPY e4e ./e4e
COPY fixtures ./fixtures
COPY legacy-route-inventory.json ./
RUN mkdir -p /app/checkfacedata && chown -R 10001:10001 /app/checkfacedata
ENV PYTHONPATH=/opt/stylegan PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
USER 10001:10001
EXPOSE 8080
CMD ["python", "legacy_checkface.py"]

FROM api-base AS runtime
COPY --from=frontend /src/deploy-self-host /app/frontend
