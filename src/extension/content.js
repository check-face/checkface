// content.js

function showPopup(hashMessage) {
  let div;
  hashMessage = hashMessage.trim();
  console.log("Face Check:", hashMessage)
  function myfunc() {
    div = document.createElement('div');
    div.innerHTML = `
      <div id="FaceCheck">
        <style>
          #FaceCheck {
            position: fixed;
            left: 0;
            right: 0;
            top: 0;
            bottom: 0;
            z-index: 9999999999999;
            display: grid;
            place-items: center;
          }

          #FaceCheck .cfcontainer {
            --text-color-normal: #333333;
            --text-color-highlight: #D65A31;
            --text-color-subtle: #999999;
            --bg-color-main: #ECECEC;
            --bg-color-alt: #CCCCCC;
          }
          
          #FaceCheck.cfdark-theme {
            --text-color-normal: #C4C6C3;
            --text-color-highlight: #D65A31;
            --text-color-subtle: #4B5365;
            --bg-color-main: #222831;
            --bg-color-alt: #21242B;
          }
          
          #FaceCheck @media (prefers-color-scheme: dark) {
            .container {
            --text-color-normal: #C4C6C3;
            --text-color-highlight: #D65A31;
            --text-color-subtle: #4B5365;
            --bg-color-main: #222831;
            --bg-color-alt: #21242B;
            }
          }
          
          #FaceCheck .checkface {
            background-color: var(--bg-color-main);
            color: var(--text-color-normal);
            padding: 10px;
            border: 2px solid var(--text-color-subtle);
          }

        </style>
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
