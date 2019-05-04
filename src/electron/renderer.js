// This file is required by the index.html file and will
// be executed in the renderer process for that window.
// All of the Node.js APIs are available in this process.
const electron = require('electron');

const mainGlobal = electron.remote.getGlobal('global');
console.log("Hashdata:", mainGlobal.hashdata);
let hashMessage = mainGlobal.hashdata;
document.querySelector("#checkfaceimg").src = "https://checkface.ml/api/" + hashMessage + "?dim=400";
if(hashMessage.length > 103) {
    hashMessage = hashMessage.substring(0, 100) + "...";
}
document.querySelector("#checkfacehash").innerHTML = hashMessage;
