<?php
// The purpose of this file is to render images for cases
// when javascript does not run e.g. social link preview bots
$value = $_GET["value"];
if(!isset($value)) { include("index.html"); }
else {
  $imgurl = "https://api.checkface.ml/api/face/?value=".rawurlencode($value); 
  $index = fopen("index.html", "r");
  while(!feof($index)) {
      $line = fgets($index);
      if (strpos($line, 'og:title') !== false) {
          ?>  <meta property="og:title" content="Check Face - <?php echo htmlspecialchars($value); ?>" \>
<?php
      }
      else if(strpos($line, '"og:image" content="assets/images/face.jpg"') !== false) {
        ?>  <meta property="og:image" content="<?php echo $imgurl."&dim=1024"; ?>" \>
<?php
      }
      else if(strpos($line, 'id="my-image"') !== false) {
        ?>        <img id="my-image" src="<?php echo $imgurl."&dim=500"; ?>" />
<?php
      }
      else {
        echo $line;
      }
  }
  fclose($index);
}
?>