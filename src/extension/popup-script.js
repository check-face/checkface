chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
    // since only one tab should be active and in the current window at once
    // the return variable should only have one entry
    var activeTab = tabs[0];

    let url = new URL(activeTab.url);
    let value = url.origin;
    let encoded = encodeURIComponent(value);


    document.getElementById("checkfaceimg").src = `https://api.checkface.ml/api/face/?value=${encoded}&dim=300`
    document.getElementById("checkfacetext").innerText = value;
});