function myFunction() {
    
    var myImage = document.getElementById("my-image");
    var value = document.getElementById('object').value;
    myImage.src = 'https://api.checkface.ml/api/face/?value=' + encodeURIComponent(value) + "&dim=500";
    window.history.pushState({ value: value }, 'Check Face - ' + value, '?value=' + encodeURIComponent(value));
    return false;
}
window.myFunction = myFunction;
const urlParams = new URLSearchParams(window.location.search);
const currentVal = (urlParams.get('value') || "");
if(currentVal) {
    document.getElementById('object').value = currentVal;
    document.getElementById("my-image").src = 'https://api.checkface.ml/api/face/?value=' + encodeURIComponent(currentVal) + "&dim=500";
}

//Can't get chrome to automatically add search engine using osdd, so get it to do a full
//form submission that counts as "doing a search" the first time in order to get it to register.
//but for subsequent searches it is better to not do a full page reload
const didRegisterSearch = urlParams.get('searchrequest') == 'get';
if(didRegisterSearch) {
    window.history.replaceState({ value: currentVal }, 'Check Face - ' + currentVal, '/?value=' + encodeURIComponent(currentVal));
    let form = document.getElementById('searchform');
    form.setAttribute("target", "_blank");
    form.setAttribute("onsubmit", "return window.myFunction()");
}

//Features appearance
let didScroll = false;
document.addEventListener('scroll', function () {
    var scroll = window.scrollY
    //>=, not <=
    if (scroll >= 500 && !didScroll) {
        didScroll = true;
        for(let el of document.getElementsByClassName("feature-icon")) { el.classList.add("feature-display") }
        for(let el of document.getElementsByClassName("feature-head-text")) { el.classList.add("feature-display") }
        for(let el of document.getElementsByClassName("feature-subtext")) { el.classList.add("feature-display") }
    }
});

for(let el of document.getElementsByClassName("add-feature-no-display")) {
    //don't hide initially incase javascript not enabled to un-hide
    el.classList.add("feature-no-display");
}