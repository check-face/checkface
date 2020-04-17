FROM python:3.7-slim as download-model
RUN pip install --no-cache-dir numpy requests
COPY ./dnnlib /app/dnnlib
COPY ./fetchModel.py /app/fetchModel.py
WORKDIR /app
RUN python fetchModel.py



FROM tensorflow/tensorflow:1.15.0-gpu-py3

RUN pip install scipy==1.3.3
RUN pip install requests==2.22.0
RUN pip install Pillow==6.2.1
RUN pip install debugpy==1.0.0b7

RUN apt update
RUN apt install --no-install-recommends -y ffmpeg
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
WORKDIR /app
COPY --from=download-model /app/.stylegan2-cache /app/.stylegan2-cache
COPY . /app
CMD ["python", "checkface.py"]