console.log('Loading function');

var http = require('http');

exports.handler = (event, context, callback) => {
  //console.log('Received event:', JSON.stringify(event, null, 2));
  var options = {
    hostname: 'worker1.checkface.ml',
    port: 443,
    path: event.path,
    method: 'GET',
    // headers: event.headers
  };

  console.log("EVENT: \n" + JSON.stringify(event, null, 2))
  
  var html = '<html><head><title>HTML from Application Loadbalancer/Lambda</title></head>' + 
        '<body><h1>HTML from API Gateway/Lambda</h1></body></html>';
  callback(null, html)
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
