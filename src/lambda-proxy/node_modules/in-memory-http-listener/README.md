# in-memory-http-listener

Overrides http server listen function to be in memory.

[![js-standard-style](https://img.shields.io/badge/code_style-standard-brightgreen.svg)](https://github.com/feross/standard)
[![build status](https://api.travis-ci.org/JamesKyburz/aws-lambda-http-server.svg)](https://travis-ci.org/JamesKyburz/aws-lambda-http-server)
[![downloads](https://img.shields.io/npm/dm/in-memory-http-listener.svg)](https://npmjs.org/package/in-memory-http-listener)
[![Greenkeeper badge](https://badges.greenkeeper.io/JamesKyburz/aws-lambda-http-server.svg)](https://greenkeeper.io/)

# usage

```javascript
const handler = require('in-memory-http-listener')

// run server code

handler(port)(req, res)

```
# license

[Apache License, Version 2.0](LICENSE)
