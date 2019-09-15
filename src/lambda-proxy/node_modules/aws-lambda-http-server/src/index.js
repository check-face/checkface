'use strict'

const httpHandler = require('in-memory-http-listener')
const createRequestResponse = require('aws-lambda-create-request-response')

module.exports = (event, context, callback) => {
  context.callbackWaitsForEmptyEventLoop =
    process.env.WAIT_FOR_EMPTY_EVENT_LOOP === 'yes'
  if (event.source === 'serverless-plugin-warmup') {
    return callback(null, 'Lambda is warm!')
  }
  const { req, res } = createRequestResponse(event, callback)
  httpHandler(process.env.SERVER_PORT || process.env.PORT)(req, res)
}
