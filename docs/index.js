'use strict';

var http = require('https');

// Handler called on Origin Request event by CloudFront.
// Copies the original URI header to a custom header field and rewrites the request to fetch /index.html as the application we are serving is a SPA.
module.exports.metaTagOriginRequestRewriter = (event, context, callback) => {
  const request = event.Records[0].cf.request;
  request.headers["x-checkface-uri"] = [{key: "X-Checkface-URI", value: request.uri}];
  console.log(JSON.stringify(request));
  request.uri = "/index.html";
  callback(null, request);
};

// Helper function used to download content from a given URL.
// As Lambda@Edge functions are very size & resource constrained we are not including request or something similar
// in the bundle but rather use the old-school way of doing things.
const downloadContent = function(url, callback) {
  http.get(url, function(res){
    var body = '';

    res.on('data', function(chunk){
      body += chunk.toString();
    });

    res.on('end', function(){
      callback(body, null);
    });
  }).on('error', function(e){
    callback(null, e);
  });
};

// Handler called on Origin Response event by CloudFront.
// Checks if we need to customize based on the originally requested URI.
// Fetches the metadata if necessary and injects it into the re-downloaded Origin document.
// Note: This is needed since due to limitations on AWS side we are not allowed to alter the response body ...
module.exports.metaTagOriginResponseRewriter = (event, context, callback) => {

  const request = event.Records[0].cf.request;
  const response = event.Records[0].cf.response;

  var uriMatches = /\?\=value=([a-z0-9]*)/gi.exec(request.headers["x-checkface-uri"][0].value);
  if (uriMatches && uriMatches.length >= 3) {
    // We have a URI match, go and load some data!
    const host = request.headers["host"][0].value;

        var metaTags = `<meta property="og:image" content="${uriMatches[0]}" />`;

        // Download the index file
        downloadContent("http://" + host + "/index.html", (indexBody, error) => {
          if (error) {
            callback(null, response);
          } else {
            var finishedBody = indexBody.replace(/(<head>)/gi, "<head>" + metaTags);

            // Generate the final response and call the callback
            const newResponse = {
              status: '200',
              statusDescription: 'OK',
              headers: response.headers,
              body: finishedBody,
            };
            callback(null, newResponse);
          }
        });

  } else {
    callback(null, response);
  }
};