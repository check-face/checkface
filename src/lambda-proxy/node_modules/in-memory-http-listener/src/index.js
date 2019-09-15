const http = require('http')

const handlers = {}

module.exports = port => handlers[port]

http.createServer = fn => {
  let port
  let handler = fn || (f => f)
  const saveHandler = fn => {
    if (fn) handler = fn
    if (typeof port !== 'undefined') handlers[port] = handler
  }
  return {
    address: () => ({ port }),
    listen (x, cb) {
      port = x
      saveHandler()
      cb && cb()
      return this
    },
    removeListener (type) {
      if (type === 'request') saveHandler(f => f)
      return this
    },
    removeAllListeners () {
      saveHandler(f => f)
      return this
    },
    on (type, fn) {
      if (type === 'request') saveHandler(fn)
      return this
    },
    addListener (type, fn) {
      this.on(type, fn)
      return this
    },
    close () {
      return this
    }
  }
}
