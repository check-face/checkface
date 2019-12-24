FROM python:3.7-slim as download-model
RUN pip install --no-cache-dir numpy==1.16.3 requests==2.21.0
COPY ./dnnlib /app/dnnlib
COPY ./fetchModel.py /app/fetchModel.py
WORKDIR /app
RUN python3.7 fetchModel.py

FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
RUN apt-get update
RUN apt-get install -y --no-install-recommends software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y --no-install-recommends python3.7 wget python3-distutils ffmpeg
RUN wget -O ~/get-pip.py https://bootstrap.pypa.io/get-pip.py
RUN python3.7 ~/get-pip.py
COPY ./requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
COPY ./dnnlib /app/dnnlib
COPY ./fetchModel.py /app/fetchModel.py
WORKDIR /app
COPY . /app
COPY --from=download-model /app/cache /app/cache
CMD ["python3.7", "checkface.py"]
