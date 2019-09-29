console.log('Loading function');

var http = require('https');

exports.handler = (event, context, callback) => {
  console.log("EVENT: \n" + JSON.stringify(event, null, 2))
  http.get('https://checkface.ml/index.html', (res) => {
    const statusCode = res.statusCode;
    const contentType = res.headers['content-type'];
    if (statusCode != 200) {
      callback(`Request Failed, status code ${statuscode}.`, null)
    }

    res.setEncoding('utf8');
    let rawData = '';
    res.on('data', (chunk) => { rawData += chunk; });
    res.on('end', () => {
      callback(null, {
        statusCode: 200,
        headers: {
          'Content-Type': contentType,
        },
        body: rawData
      });
    });
  }).on('error', (e) => {
    callback(`Got error: ${e.message}`, null);
  });
};
