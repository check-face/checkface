// background.js
var contextMenuItem={
  "id": "checkFace",
  "title": "Check Face \“%s\”",
  "contexts": ["selection"]
};

var fileIdHash = "";

chrome.contextMenus.create(contextMenuItem);
// Called when the user clicks on the browser action.
chrome.contextMenus.onClicked.addListener(function(info, tab) {
  // Send a message to the active tab
  chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
    var activeTab = tabs[0];
    chrome.tabs.sendMessage(activeTab.id, {"message": info.selectionText });
  });
});

chrome.tabs.sendMessage(0, {"message": "asdfggh" });


String.prototype.hashCode = function() {
var hash = 0, i, chr;
if (this.length === 0) return hash;
for (i = 0; i < this.length; i++) {
  chr   = this.charCodeAt(i);
  hash  = ((hash << 5) - hash) + chr;
  hash |= 0; // Convert to 32bit integer
}
return hash;
};
