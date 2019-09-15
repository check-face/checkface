console.log('Loading function');

var http = require('https');

exports.handler = (event, context, callback) => {
  //console.log('Received event:', JSON.stringify(event, null, 2));
  var options = {
    hostname: 'localhost',
    port: 443,
    path: event.path,
    method: 'GET',
    // headers: event.headers
  };

  console.log("EVENT: \n" + JSON.stringify(event, null, 2))

  http.request(options, function(res) {
    console.log("Got response: " + res.statusCode);
    body = '';
    res.on("data", function(chunk) {
      body += chunk;
    });
    res.on("end", function() {
      console.log("BODY: " + chunk);
      callback(null, {
        statusCode: res.statusCode,
        headers: res.headers,
        body: body
      });
    });
  }).on('error', function(e) {
    console.log("Got error: " + e.message);
  });

  
  // return new Promise(function(resolve, reject) {
  //   resolve(200);
  // var req = http.request(options, res => {

  //   res.on('data', d => {
  //     resolve(200);
  //   })
  // });
  // req.end()
  // })
  // .then (()=>{
  //       return {
  //           contentType: 'text/html',
  //           statusCode : 200
  //       }
  //   });
};
