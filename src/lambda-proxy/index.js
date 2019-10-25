console.log('Loading function');

var http = require('https');

exports.handler = (event, context, callback) => {
  process.env["NODE_TLS_REJECT_UNAUTHORIZED"] = 0;
  //console.log('Received event:', JSON.stringify(event, null, 2));
  
  //console.log("EVENT: \n" + JSON.stringify(event, null, 2));

  var options = {
    hostname: 'worker1.checkface.ml',
    port: 443,
    path: "/api/asdf",
    method: 'GET',
    // headers: event.headers
  };
  let url = "https://" + options.hostname + options.path;
  //console.log("options: \n" + JSON.stringify(options, null, 2))
  http.get(url, function(res) {
    console.log("Got response: " + JSON.stringify(res.headers));
    let body = new Buffer("");
    res.on("data", function(chunk) {
      body = Buffer.concat([body, chunk]);
    });
    
    res.on("end", function() {
      console.log("BODY: \n" + JSON.stringify(body));
      callback(null, {
        statusCode: res.statusCode,
        headers: {"Content-Type": "image/jpg; charset=utf-8"},
        image: body.toString("base64"),
        isBase64Encoded: true
      });
    });
  }).on('error', function(e) {
    console.log("Got error: " + e.message);
  });
};
