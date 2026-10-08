"""encoder4editing image -> W+ latent server, called by the API as http://encoderapi:8080/api/encodeimage/.

Rebuild of check-face/encoder4editing scripts/server_inference.py (the deployed olicoad/encoder4editing image, which
is no longer on Docker Hub). Same HTTP contract, queue/batch worker, alignment fallback and preprocessing. It loads
only the e4e encoder plus latent_avg from the official e4e_ffhq_encode.pt checkpoint: the deployed server built the
whole pSp net but only ever called net.encoder, so the StyleGAN decoder and its JIT-compiled CUDA ops are not needed.

POST /api/encodeimage/  form: usrimg=<image file>, tryalign=true|false
  -> {"dlatent": [[...512 floats] x 18], "did_align": bool}
"""
import argparse
import os
import queue
import threading
from types import SimpleNamespace

import dlib
import flask
import numpy as np
import PIL.Image
import torch

from e4e.models.encoders.psp_encoders import Encoder4Editing
from e4e.utils.alignment import align_face, NumberOfFacesError

MODELS = os.environ.get('ENCODER_MODEL_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pretrained_models'))
PIL.Image.MAX_IMAGE_PIXELS = 8192 * 4096  # protect against decompression bomb DOS attacks

app = flask.Flask(__name__)
job_queue = queue.Queue()
dlib_shape_predictor = None
device = None


class EncodeJob:
    def __init__(self, img, name, align):
        self.img = img
        self.align = align
        self.name = name
        self.evt = threading.Event()
        self.did_align = False
        self.latent = None

    def __str__(self):
        return self.name

    def set_result(self, latent):
        self.latent = latent
        self.evt.set()

    def wait_for_result(self, timeout):
        if self.evt.wait(timeout):
            return self.latent
        return None


@app.route('/api/encodeimage/', methods=['POST'])
def encode_image():
    tryalign = flask.request.form.get('tryalign', 'false')
    do_align = tryalign.lower() == 'true'
    file = flask.request.files.get('usrimg')
    if not file:
        return flask.Response('No file uploaded for usrimg', status=400)

    img = PIL.Image.open(file.stream)
    img = img.convert('RGB')  # processing steps are expecting an RGB image
    job = EncodeJob(img, 'user uploaded file', do_align)
    job_queue.put(job)
    latent = job.wait_for_result(int(os.environ.get('ENCODER_TIMEOUT', '15')))
    if latent is None:
        raise Exception('Encoding image failed or timed out')
    return flask.jsonify({'dlatent': latent.tolist(), 'did_align': job.did_align == True})


def run_alignment(img):
    try:
        aligned_image = align_face(filepath=None, predictor=dlib_shape_predictor, img=img)
        print('Aligned image has shape: {}'.format(aligned_image.size))
        return aligned_image
    except NumberOfFacesError:
        return None


def transform_inference(img):
    # == torchvision transforms.Compose([Resize((256, 256)), ToTensor(), Normalize([0.5]*3, [0.5]*3)]) on a PIL
    # image: torchvision's Resize calls PIL's bilinear resize for PIL inputs.
    pixels = np.asarray(img.resize((256, 256), PIL.Image.BILINEAR), dtype=np.float32) / 255.0
    return (torch.from_numpy(pixels).permute(2, 0, 1) - 0.5) / 0.5


def load_encoder(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)  # pinned official checkpoint, sha256-checked at build
    opts = dict(ckpt['opts'])
    opts['stylegan_size'] = 1024  # what setup_model() sets for the ffhq checkpoint
    opts = SimpleNamespace(**opts)
    net = Encoder4Editing(50, 'ir_se', opts)
    net.load_state_dict({k[len('encoder.'):]: v for k, v in ckpt['state_dict'].items() if k.startswith('encoder.')}, strict=True)
    net.eval().to(device)
    latent_avg = ckpt['latent_avg'].to(device) if opts.start_from_latent_avg else None
    return net, latent_avg


def get_latents(net, latent_avg, x):
    codes = net(x)
    if latent_avg is not None:
        if codes.ndim == 2:
            codes = codes + latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        else:
            codes = codes + latent_avg.repeat(codes.shape[0], 1, 1)
    return codes


def get_batch(batchsize):
    yield job_queue.get(True)  # will block until it gets a job
    for _ in range(batchsize - 1):
        if not job_queue.empty():
            yield job_queue.get_nowait()


def worker(ckpt, batch_size):
    net, latent_avg = load_encoder(ckpt)
    print(f'Worker ready on {device}', flush=True)
    while True:
        jobs = list(get_batch(batch_size))
        app.logger.info(f'Running jobs {[str(job) for job in jobs]}')
        try:
            imgs = []
            for job in jobs:
                aligned = run_alignment(job.img) if job.align else None
                if aligned is None:
                    imgs.append(job.img)
                else:
                    imgs.append(aligned)
                    job.did_align = True
            batch = torch.stack([transform_inference(img) for img in imgs])
            with torch.no_grad():
                latents = get_latents(net, latent_avg, batch.to(device).float()).cpu()
            for latent, job in zip(latents, jobs):
                job.set_result(np.array(latent))
        except Exception:
            app.logger.exception('Encoding batch failed')
            for job in jobs:
                job.set_result(None)
        app.logger.info('Finished batch job')


def pick_device(requested):
    if requested != 'auto':
        return requested
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        archs = torch.cuda.get_arch_list()
        # SASS for an older minor version of the same major runs (sm_60 on sm_61); otherwise fall back to CPU.
        if any(a.startswith('sm_') and int(a[3:]) // 10 == major and int(a[3:]) % 10 <= minor for a in archs):
            return 'cuda'
        print(f'GPU sm_{major}{minor} is not in this torch build ({archs}); encoding on CPU', flush=True)
    return 'cpu'


def main(args):
    global dlib_shape_predictor, device
    device = pick_device(args.device)
    dlib_shape_predictor = dlib.shape_predictor(os.path.join(MODELS, 'shape_predictor_68_face_landmarks.dat'))
    t1 = threading.Thread(target=worker, args=[args.ckpt, args.batch])
    t1.daemon = True  # kill thread on program termination (to allow keyboard interrupt)
    t1.start()

    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # limit to 16mb
    app.run(host='0.0.0.0', port=args.api_port, threaded=True)
    app.logger.info('Closing server_inference')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a flask server to do e4e inference on images')
    parser.add_argument('--batch', type=int, default=1, metavar='BATCH_SIZE', help='batch size for the encoder')
    parser.add_argument('--api-port', type=int, default=8080, metavar='PORT', help='port to listen on (default 8080)')
    parser.add_argument('--ckpt', default=os.path.join(MODELS, 'e4e_ffhq_encode.pt'), metavar='CHECKPOINT', help='path to e4e checkpoint')
    parser.add_argument('--device', default=os.environ.get('ENCODER_DEVICE', 'auto'), help='auto (default), cuda or cpu')
    main(parser.parse_args())
