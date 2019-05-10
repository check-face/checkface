$(document).ready(function () {

    var myVideo = document.getElementById("video1");

    function playPause() {
        if (myVideo.paused)
            myVideo.play();
        else
            myVideo.pause();
    }

    function makeBig() {
        myVideo.width = 560;
    }

    function makeSmall() {
        myVideo.width = 320;
    }

    function makeNormal() {
        myVideo.width = 420;
    }

    function myFunction() {
       
        var myImage = document.getElementById("my-image");
        myImage.src = 'https://api.checkface.ml/api/' + document.getElementById('object').value + "?dim=500";


        // document.getElementById("imagecontainer").appendChild(myImage);
        return false;
    }
    window.myFunction = myFunction;


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
