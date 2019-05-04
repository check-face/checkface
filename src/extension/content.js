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
            <p>sasfdfshdllfdaflafdsja</p>
            <img id="checkfaceimg" width=300 height=300 />
          </div>
        </div>      
      </div>
    `

    document.body.appendChild(div);
    div.onclick = removeFunc;
    document.querySelector("#checkfaceimg").src = "http://checkface.ml/api/" + hashMessage + "?dim=300";
  }

  function removeFunc() {
    div.remove();
    return true;
  }
  $(document).ready(myfunc);
}
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => showPopup(request.message));