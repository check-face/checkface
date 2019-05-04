# FROM ufoym/deepo:tensorflow-py36-cu100
FROM ufoym/deepo
# RUN pip install ipython pillow psutil requests tensorflow==1.13.1 tqdm keras
RUN pip uninstall -y tf-nightly-gpu-2.0-preview
RUN pip install tensorflow-gpu

COPY ./dnnlib /app/dnnlib
COPY ./fetchModel.py /app/fetchModel.py
WORKDIR /app
RUN python fetchModel.py

COPY . /app
CMD ["python", "checkface.py"]