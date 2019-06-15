$(document).ready(function () {
    function getImage(seed) {

        //generate normal distribution random number
        function randn_bm() {
            let u = 0, v = 0;
            while(u === 0) u = Math.random(); //Converting [0,1) to (0,1)
            while(v === 0) v = Math.random();
            return Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
        }

        function genVec(seed) {
            Math.seedrandom(seed)
            let arr = []
            for(let i = 0; i < 512; i++) {
                arr.push(randn_bm())
            }
            return arr
        }

        const inputs = {
            "z": genVec(seed),
            "truncation": 0.7
        };

        return fetch('http://localhost:8001/query', {
            method: 'POST',
            headers: {
                Accept: 'application/json',
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(inputs)
        })
        .then(response => response.json())
        .then(output => {
            const { image } = output;
            return image
        })
    }
    function myFunction() {
       
        var myImage = document.getElementById("my-image");
        var value = document.getElementById('object').value;
        // myImage.src = 'https://api.checkface.ml/api/' + value + "?dim=500";
        getImage(value).then(image => {
            myImage.src = image;
        })

        window.history.replaceState({ value: value }, 'Check Face - ' + value, '?value=' + value);
        return false;
    }
    window.myFunction = myFunction;
    const urlParams = new URLSearchParams(window.location.search);
    const currentVal = urlParams.get('value');
    if(currentVal) {
        document.getElementById('object').value = currentVal;
        // document.getElementById("my-image").src = 'https://api.checkface.ml/api/' + currentVal + "?dim=500";
        getImage(currentVal).then(image => {
            document.getElementById("my-image").src = image;
        })
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
