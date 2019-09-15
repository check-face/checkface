# aws-lambda-http-server

Call your http server stack code using an in memory http listener. No sockets needed.

[![js-standard-style](https://img.shields.io/badge/code_style-standard-brightgreen.svg)](https://github.com/feross/standard)
[![build status](https://api.travis-ci.org/JamesKyburz/aws-lambda-http-server.svg)](https://travis-ci.org/JamesKyburz/aws-lambda-http-server)
[![downloads](https://img.shields.io/npm/dm/aws-lambda-http-server.svg)](https://npmjs.org/package/aws-lambda-http-server)
[![Greenkeeper badge](https://badges.greenkeeper.io/JamesKyburz/aws-lambda-http-server.svg)](https://greenkeeper.io/)

## server.js

```javascript
require('http').createServer((req, res) => {
  if (req.url === '/hello') return res.end('world')
})
.listen(5000)
```

## aws-lambda.js

```javascript
exports.proxy = require('aws-lambda-http-server')
require('./server.js')
```

## serverless.yml

```yaml
service: test
provider:
  name: aws
  runtime: nodejs8.10
  endpointType: edge
  region: eu-west-1
functions:
  proxy:
    handler: aws-lambda.proxy
    environment:
      PORT: 5000
      BINARY_SUPPORT: 'no'
    events:
      - http:
          path: /{proxy+}
          method: any
      - http:
          path: ''
          method: any
```

# license

[Apache License, Version 2.0](LICENSE)
