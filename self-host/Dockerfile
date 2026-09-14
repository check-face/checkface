FROM python:3.11-slim-bookworm
ARG STYLEGAN_REV=d72cc7d041b42ec8e806021a205ed9349f87c6a4
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git /opt/stylegan && cd /opt/stylegan && git checkout "$STYLEGAN_REV" && rm -rf .git
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements.txt
COPY api.py smoke.py prepare_model.py test_transport.py ./
COPY fixtures ./fixtures
ENV PYTHONPATH=/opt/stylegan PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
USER 10001:10001
EXPOSE 8080
CMD ["python", "api.py"]
