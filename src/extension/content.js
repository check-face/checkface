// content.js
chrome.runtime.onMessage.addListener(

  function(request, sender, sendResponse) {
      let div;
      function myfunc(){

        div = document.createElement( 'div' );
        div_below = document.createElement('div');
        var img = document.createElement('img');

        document.body.appendChild( div );
        div.appendChild( div_below );
        div.onclick = removeFunc;
        div.style.position = 'absolute';
        div.style.left = '0';
        div.style.top = '0';
        div.style.bottom = '0';
        div.style.right = '0';
        div_below.style.margin = "auto";
        div.style.zIndex = '99999999999';
        div_below.style.width = '500px';
        div_below.style.height = '5000px';
        div_below.style.border = '2px solid black';
        div_below.style.backgroundColor = 'white';
        div_below.appendChild( img ).src = 'https://picsum.photos/300/300';
        img.style.position = 'absolute';
        img.style.left = '0';
        img.style.top = '0';
        img.style.bottom = '0';
        img.style.right = '0';
        img.style.margin = "auto";
      }

      function removeFunc(){
        div.remove();
        return true;
      }
      $(document).ready(myfunc);

    }
);
