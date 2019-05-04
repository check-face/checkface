// content.js
chrome.runtime.onMessage.addListener(
  function(request, sender, sendResponse) {
    if( request.message === "clicked_browser_action" ) {
      function myfunc(){
        document.documentElement.style.height = '100%';
        document.body.style.height = '100%';
        document.documentElement.style.width = '100%';
        document.body.style.width = '100%';

        var div = document.createElement( 'div' );
        var head = document.createElement( 'head');
        var link = document.createElement('link');
        var div_below = document.createElement( 'div' );
        var p = document.createElement('p');
        var img = document.createElement('img');
        // var cssId = 'myCss';  // you could encode the css path itself to generate id..
        // if (!document.getElementById(cssId))
        // {
        //     var head  = document.getElementsByTagName('head')[0];
        //     var link  = document.createElement('link');
        //     link.id   = cssId;
        //     link.rel  = 'stylesheet';
        //     link.type = 'text/css';
        //     link.href = '../shared/styles.css';
        //     link.media = 'all';
        //     head.appendChild(link);
        // }

        document.body.appendChild( div );

        div.class = 'container dark-theme';
        div.style.position = 'absolute';
        div.style.display = 'block';
        div.style.zIndex = '1000';
        div.style.width = '50%';
        div.style.border = '5px solid red';
        div.style.backgroundColor = 'white';
        div.appendChild( div_below ).class = 'checkface';
        div_below.style.zIndex = '1000';
        div_below.appendChild(p).innerHTML = 'sasfdfshdllfdaflafdsja';
        div_below.appendChild( img ).src = 'https://picsum.photos/300/300';

  }

  $(document).ready(myfunc);

      // This line is new!
      // chrome.runtime.sendMessage({"message": "open_popup", "url": myfunc});
    }
  }
);
