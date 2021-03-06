# Check Face

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/cfdfc39a457c4a68a1fce72b3f357a86)](https://app.codacy.com/app/cdilga/checkface?utm_source=github.com&utm_medium=referral&utm_content=check-face/checkface&utm_campaign=Badge_Grade_Dashboard)

 > Putting a face to a hash

`Winner of Facebook Hack Melbourne 2019`


Facebook's [Hackathon](https://www.facebook.com/hackathon/) at [Facebook Hack Melbourne 2019](https://www.facebook.com/events/408587233300017/)

Who uses checksums? We all know we should.

A range of unused tools exist for verifying file integrity that suffer from poor adoption, are difficult to use and aren't human-friendly.
Humans are inherently good at remembering interesting information, be it stories, people and generally benefit from context. Most humans also have the ability to remember faces extremely well, with many of us experiencing false-positives or pareidolia - seeing faces as a part of inanimate objects.

With the advent of hyper-realistic Style transfer GAN's like [Nvidia's StyleGAN2](https://github.com/NVlabs/stylegan2), we can generate something that our brains believe is a real person, and make use of that human-hardware accelerated memorisation and let people compare between hashes they've seen, potentially even weeks apart, with only a few quick glances.


![CheckFace Face](/docs/assets/images/face.jpg)  
*This generated face is an example of what you could expect to see next to your file's checksum or your git commit sha.*

## Our Stack
   - [Nvidia StyleGAN2](https://github.com/NVlabs/stylegan2)
     - Tensorflow
   - Docker
     - Nvidia Docker runtime
   - Flask
   - GitHub Pages
   - Chrome Web Extension
   - CloudFlare


## Quickstart

 - **Chrome Extension** Context Menu
 - **Backend API** running a Dockerized Nvidia Stylegan2 on Flask
 - **Project Webpage**

### Chrome Extension

The `/src/extension` directory holding the manifest file can be added as an extension in developer mode in its current state.

Open the Extension Management page by navigating to [chrome://extensions](chrome://extensions).
The Extension Management page can also be opened by clicking on the Chrome menu, hovering over More Tools then selecting Extensions.
Enable Developer Mode by clicking the toggle switch next to Developer mode.
Click the LOAD UNPACKED button and select the extension directory.

![How to load extension in chrome with developer mode](https://developer.chrome.com/static/images/get_started/load_extension.png)


#### Load Extension

Ta-da! The extension has been successfully installed. Because no icons were included in the manifest, a generic toolbar icon will be created for the extension.

(Sourced: [Chrome Developer](https://developer.chrome.com/extensions/getstarted))

### Backend API

Request images at [api.facemorph.me/api/face?value=example&dim=300](https://api.facemorph.me/api/face?value=example&dim=300).

Documentation for endpoints available on api is available at https://checkface.facemorph.me/api

Prerequisites to run the backend server

  - Nvidia GPU with sufficient VRAM to hold the model (recommended atleast 6 GB)
  - Nvidia drivers (eg. `sudo apt install nvidia-driver-435`)
  - [Nvidia Docker runtime](https://github.com/NVIDIA/nvidia-docker) (only supported on Linux, until HyperV adds GPU passthrough support)

If you don't have a suitable GPU, you canrun the backend with an AWS p3 instance on ECS, or g3s.xlarge via docker-machine for testing.

The backend API is intented to be used behind a reverse proxy and trusts 1 `X-Forwarded-For` [using proxy_fix middleware](https://werkzeug.palletsprojects.com/en/1.0.x/middleware/proxy_fix/))

#### Build and run docker image
```console
cd ./src/server
docker build -t checkfaceapi .
docker run -d -p 8080:8080 -p 8000:8000 --gpus all --name checkfaceapi checkfaceapi
```

The api is available on port 8080 and prometheus metrics on port 8000.

You may want to mount `/app/checkfacedata` to save generated media.
#### Environment variables

 - `LOW_GPU_MEM` defaults to `false`. Set `true` to configure tf gpu options to work with less memory.
 - `GENERATOR_BATCH_SIZE` defaults to `10`. Set to `4` or lower if running with less GPU mem or a lower end system.
 - `MONGODB_CONNECTION_STRING` to override the connection string for mongodb.

#### MongoDB
MongoDB is needed for features such as using guids with uploaded image or latents. The default connection string is set for use with a container named `db` in a docker network.

You can override the connection string using the environment variable.

If using docker, note that you can't use docker-compose with nvidia GPUs. The compose file is only for reference. You will have to manually connect the checkface server to mongodb using a docker network.

#### Image encoding
Image encoding is currently done with a seperate internal API based on open source repos. We plan on merging it into one process rather than as a seperate container if possible. In the mean time, feel free to contact us or open an issue if you would like more information on how we currently do it.

### Project Webpage

Simple pure Javascript based bootstrap webpage. Upload to anything that serves static files

## Development

### Chrome Extension

1. Open `chrome://extensions`
2. Enable "Developer mode"
3. Load unpacked and select the folder `src/extension`

### Backend API

We rely on Nvlabs StyleGAN2 to run our inference, using the default model.
You can run it in docker to avoid all the dependencies, as shown above.

#### System requirements

All you really need is a CUDA GPU with enough VRAM to load the inference model, which has been tested to work on a RTX 2080 Ti with 11GB of VRAM, with NVIDIA driver 435.21.

## License

Our work is based on a combination of original content and work adapted from [Nvidia Labs StyleGAN2](https://github.com/NVlabs/stylegan2) under the [Nvidia Source Code License-NC](https://github.com/NVlabs/stylegan2/blob/master/LICENSE.txt). Anything outside of the `src/server` dir is original work, and a diff can be used to show the use of the dnnlib and StyleGAN model inside of this directory.

The inference model was trained by Nvidia Labs on the FFHQ dataset, please refer to the [Flickr-Faces-HQ repository](http://stylegan.xyz/ffhq).
