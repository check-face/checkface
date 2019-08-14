// content.js

function showPopup(hashMessage) {
  let div;
  hashMessage = hashMessage.trim();
  console.log("Face Check:", hashMessage)
  function myfunc() {
    div = document.createElement('div');
    div.innerHTML = `
      <div id="FaceCheck">
        <div class="cfcontainer">
          <div class="checkface">
            <p id="checkfacehash">sasfdfshdllfdaflafdsja</p>
            <img id="checkfaceimg" width=300 height=300 />
          </div>
        </div>      
      </div>
    `

    document.body.appendChild(div);
    div.onclick = removeFunc;
    div.querySelector("#checkfaceimg").src = "https://api.checkface.ml/api/face/?value=" + hashMessage + "&dim=300";
    if(hashMessage.length > 103) {
      hashMessage = hashMessage.substring(0, 100) + "...";
    }
    div.querySelector("#checkfacehash").innerHTML = hashMessage;

  }

  function removeFunc() {
    div.remove();
    return true;
  }
  $(document).ready(myfunc);
}
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => showPopup(request.message));

let commitSHA_Regex = /(^| |\/|=)([a-zA-Z0-9]{40,40})($|\?|#|&|\/| )/;

function addHoverFaceCheck(node) {
  if(node.querySelector("#cfToolTip")) {
    return;
  }
  let res = commitSHA_Regex.exec(node.href)
  if(res) {
    hashMessage = res[2]
    // node.style.color = 'yellow';
    var div = document.createElement('div');
    div.id = "cfToolTip";
    div.innerHTML = `
    <div class="cfcontainer">
      <div class="checkface">
        <p id="checkfacehash">sasfdfshdllfdaflafdsja</p>
        <img id="checkfaceimg" width=200 height=200 />
      </div>
    </div>
    `
    div.querySelector("#checkfaceimg").src = "https://api.checkface.ml/api/face/?value=" + hashMessage + "&dim=200";
    div.querySelector("#checkfacehash").innerHTML = hashMessage.substring(0, 7);
    node.appendChild(div);
  }
}

function observeAllLinks() {
  let selector = 'a[href*=commit]';
  // Initialize results with current nodes.
  var result = Array.prototype.slice.call(document.querySelectorAll(selector));
  result.forEach(addHoverFaceCheck);
  // Create observer instance.
  var observer = new MutationObserver(function(mutations) {
    mutations.forEach(function(mutation) {
      console.log("got mutation", mutation);
      [].forEach.call(mutation.addedNodes, function(node) {
        if (node.nodeType === Node.ELEMENT_NODE) {
          if(node.matches("#cfToolTip") || node.matches("#FaceCheck")) {
            return;
          }
          var res = Array.prototype.slice.call(document.querySelectorAll(selector));
          res.forEach(addHoverFaceCheck);
        }
      });
    });
  });

  // Set up observer.
  observer.observe(document, { childList: true, subtree: true });

  return result;
}


$(document).ready(() => {
  observeAllLinks()
});