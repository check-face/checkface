var tags = document.querySelectorAll("a[class*='sha'],a[href^='\/commit\/']");
var i;
for(i = 0; i < tags.length; i++)
{
    tags[i].onmouseover = function(){
      alert(tags.length);
    }

}
