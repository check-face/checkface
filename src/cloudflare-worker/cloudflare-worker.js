addEventListener('fetch', event => {
    event.respondWith(handleRequest(event.request))
  })
  
  class MetaTagElementHandler {
    constructor(value) {
      this.value = value
    }
    element(element) {
      let prop = element.getAttribute("property")
      switch(prop) {
        case "og:image": {
          let imgUrl = `https://api.checkface.ml/api/face/?value=${encodeURIComponent(this.value)}&dim=1024`
          element.setAttribute('content', imgUrl)
          break;
        }
        case "og:title": {
          let title = "Check Face - " + this.value
          element.setAttribute('content', title)
          break;
        }
      }
    }
  }
  
  class ImgElementHandler {
    constructor(value) {
      this.value = value
    }
    element(element) {
        let imgUrl = `https://api.checkface.ml/api/face/?value=${encodeURIComponent(this.value)}&dim=500`
        element.setAttribute('src', imgUrl)
    }
  }
  
  function replaceUrl(request, url) {
    const bodyP = request.headers.get('Content-Type') ? request.blob() : Promise.resolve(undefined);
    const newRequestP =
    bodyP.then((body) =>
      new Request(url, {
        method: request.method,
        headers: request.headers,
        body: body,
        referrer: request.referrer,
        referrerPolicy: request.referrerPolicy,
        redirect: request.redirect
      })
    );
    return newRequestP
  }
  
  class InputElementHandler {
    constructor(value) {
      this.value = value
    }
    element(element) {
      element.setAttribute('value', this.value)
    }
  }
  
  function replaceUrl(request, url) {
    const bodyP = request.headers.get('Content-Type') ? request.blob() : Promise.resolve(undefined);
    const newRequestP =
    bodyP.then((body) =>
      new Request(url, {
        method: request.method,
        headers: request.headers,
        body: body,
        referrer: request.referrer,
        referrerPolicy: request.referrerPolicy,
        redirect: request.redirect
      })
    );
    return newRequestP
  }
  
  async function handleRequest(req) {
    //return await fetch("https://olisworld.com")
    // console.log("serving request for " + req.url)
    let url = new URL(req.url)
    url.hostname = "checkface.ml"
    req = await replaceUrl(req, url.toString())
    let res = await fetch(req)
    // if(url.pathname = '/') {
    let value = url.searchParams.get("value")
    if(!value) {
      value = ""
    }
    if(url.pathname == "/" || url.pathname == "/index.html") {
      console.log(`transforming meta tags of ${url} for value=${value}`)
      res = new HTMLRewriter()
        .on("meta", new MetaTagElementHandler(value))
        .on("img#my-image", new ImgElementHandler(value))
        .on("form#searchform>input#object", new InputElementHandler(value))
        .transform(res)
    }
    return res
  }