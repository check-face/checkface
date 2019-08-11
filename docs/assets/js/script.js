$(document).ready(function () {

    function myFunction() {
       
        var myImage = document.getElementById("my-image");
        var value = document.getElementById('object').value;
        myImage.src = 'https://api.checkface.ml/api/' + encodeURIComponent(value) + "?dim=500";
        window.history.pushState({ value: value }, 'Check Face - ' + value, '?value=' + encodeURIComponent(value));
        return false;
    }
    window.myFunction = myFunction;
    const urlParams = new URLSearchParams(window.location.search);
    const currentVal = decodeURIComponent(urlParams.get('value') || "");
    if(currentVal) {
        document.getElementById('object').value = currentVal;
        document.getElementById("my-image").src = 'https://api.checkface.ml/api/' + encodeURIComponent(currentVal) + "?dim=500";
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

    //Mouse click scroll
    $(document).ready(function () {
        $(".mouse").click(function () {
            $('html, body').animate({scrollTop: '+=750px'}, 1200);
        });
    });

    //Features appearance
    $(window).scroll(function () {
        var scroll = $(window).scrollTop();

        //>=, not <=
        if (scroll >= 500) {
            $(".feature-icon").addClass("feature-display");
            $(".feature-head-text").addClass("feature-display");
            $(".feature-subtext").addClass("feature-display");
        }
    });


//smooth scrolling

    $('a[href*="#"]:not([href="#"])').click(function () {
        if (location.pathname.replace(/^\//, '') == this.pathname.replace(/^\//, '') && location.hostname == this.hostname) {
            var target = $(this.hash);
            target = target.length ? target : $('[name=' + this.hash.slice(1) + ']');
            if (target.length) {
                $('html, body').animate({
                    scrollTop: target.offset().top
                }, 1000);
                return false;
            }
        }
    });


})
