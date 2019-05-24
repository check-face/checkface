// This file is required by the index.html file and will
// be executed in the renderer process for that window.
// All of the Node.js APIs are available in this process.
const electron = require('electron');

const mainGlobal = electron.remote.getGlobal('global');
let hashdata = mainGlobal.hashdata;
console.log("Hashdata:", hashdata);

let img = document.querySelector("#checkfaceimg");
let siteurl = `https://checkface.ml/`;
if(hashdata.sum) {
    img.src = `https://api.checkface.ml/api/${hashdata.sum}?dim=400`;
    siteurl += `?value=${hashdata.sum}`
}
else {
    img.classList.add("dummyimg");
}
img.addEventListener('click', () => electron.shell.openExternal(siteurl));
let hashMessage = hashdata.message;
if(hashMessage.length > 103) {
    hashMessage = hashMessage.substring(0, 100) + "...";
}
document.querySelector("#checkfacehash").innerHTML = hashMessage;
