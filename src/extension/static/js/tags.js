var tags = document.querySelectorAll("a[href^='\/commit'], a[class*='sha']");
var i;
for(i = 0; i < tags.length; i++)
{
    tags[i].onmouseover = function(){
      alert(tags.length);

    }

}
