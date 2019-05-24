$(document).ready(function () {

    function myFunction() {
       
        var myImage = document.getElementById("my-image");
        var value = document.getElementById('object').value;
        myImage.src = 'https://api.checkface.ml/api/' + value + "?dim=500";
        window.history.replaceState({ value: value }, 'Check Face - ' + value, '?value=' + value);
        return false;
    }
    window.myFunction = myFunction;
    const urlParams = new URLSearchParams(window.location.search);
    const currentVal = urlParams.get('value');
    if(currentVal) {
        document.getElementById('object').value = currentVal;
        document.getElementById("my-image").src = 'https://api.checkface.ml/api/' + currentVal + "?dim=500";
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

    //Subscribe to newsletter
    /*$('#object').on('submit', function (event) {
      alert("wdldf");
      event.preventDefault();
        const form = event.target;
        const data = new FormData(form);

        var myImage = new Image();
        myImage.src = 'checkface.ml/api/'+data.object.value;


        document.getElementById("imagecontainer").appendChild(myImage);

        return true;
        });*/


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
