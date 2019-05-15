# CheckFace

![CheckFace Face](/docs/assets/images/face.jpg)

 > Putting a face to a hash

`Winner of Facebook Hack Melbourne 2019`


Facebook's [Hackathon](https://www.facebook.com/hackathon/) at [Facebook Hack Melbourne 2019](https://www.facebook.com/events/408587233300017/)

Who uses checksums? We all know we should.

A range of unused tools exist for verifying file integrity that suffer from poor adoption, are difficult to use and aren't human-friendly.
Humans are inherently good at remembering interesting information, be it stories, people and generally benefit from context. Most humans also have the ability to remember faces extremely well, with many of us experiencing false-positives or pareidolia - seeing faces as a part of inanimate objects.

With the advent of hyper-realistic Style transfer GAN's like Nvidia's StyleGAN, we can generate something that our brains believe is a real person, and make use of that human-hardware accelerated memorisation and let people compare between hashes they've seen even weeks apart with only a few quick glances.

## Our Stack
   - [Nvidia StyleGAN](https://stylegan.xyz/code)
     - Tensorflow
   - Docker
     - Nvidia Docker runtime
   - Flask
   - Amazon AWS
     - ECR
     - S3 Hosted Site [Checkface](http://checkface.ml)
   - Chrome Web Application
   - Electron Application
   - CloudFlare


## Quickstart

 - **Chrome Extension** Context Menu
 - **Electron App** Context Menu
 - **Backend API** running a Dockerized Nvidia Stylegan on Flask
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
### Electron App

### Backend API

Prerequisites

  - GPU with sufficient VRAM to hold the model
  - Nvidia Docker runtime (only supported on Linux, until HyperV adds GPU passthrough support)

For running a backend we have used an AWS p3 instance on ECS, or g3s.xlarge via docker-machine for testing.

### Project Webpage

Simple pure Javascript based bootstrap webpage. Upload to anything that serves static files

## Development

### Chrome Extension

TODO

### Electron App

TODO

### Backend API

TODO
