// background.js
var contextMenuItem={
  "id": "checkFace",
  "title": "Check Face \“%s\”",
  "contexts": ["selection"]
};

chrome.contextMenus.create(contextMenuItem);
// Called when the user clicks on the browser action.
chrome.contextMenus.onClicked.addListener(function(tab) {
  // Send a message to the active tab
  console.log("hello");
  chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
    var activeTab = tabs[0];
    chrome.tabs.sendMessage(activeTab.id, {"message": "clicked_browser_action"});
  });
});
