// This file is required by the index.html file and will
// be executed in the renderer process for that window.
// All of the Node.js APIs are available in this process.
const electron = require('electron');

const mainGlobal = electron.remote.getGlobal('global');
console.log(mainGlobal.hashdata);
let hashMessage = mainGlobal.hashdata;
document.querySelector("#checkfaceimg").src = "https://checkface.ml/api/" + hashMessage + "?dim=300";
document.querySelector("#checkfacehash").innerHTML = hashMessage;
