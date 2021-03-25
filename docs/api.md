---
title: API Docs
layout: default
---

<div>
    <a class="checkface-version-badge" id="page-version-checkface-link">
        <div class="text">
            <p>Checkface for this page</p>
        </div>
        <div class="img">
            <img id="page-version-checkface-img" alt="CheckFace based on hash for this page"/>
        </div>
    </a>
</div>
--------


# Using the CheckFace API

The API is available at `https://api.checkface.ml/`

You're in luck that it's not authenticated; it's an entirely open API!

The API is in Alpha is is likely to go down, have maintenance at unpredictable times and make breaking changes to the API.


# Endpoints

## /status/
Returns a http 200 success if the API is available

## /api/face/

Generate a single image
Use either `value`, `seed` or `guid` parameters to specify the face

### Query Parameters

`dim` *optional* **int** between 10-1024

`value` *optional* **string** is any free text such as the hash of a file 

`seed` *optional* **int** corresponding to the seed used in the random number generator to generate latents

`guid` *optional* **guid** globally unique identifier returned by [/api/registerlatent](#post-apiregisterlatent)

`format` *optional* **string** if set to webp, returns a webp image. Defaults to jpg

### Returns

An image

### Example

[https://api.checkface.ml/api/face/?value=example&dim=500&format=webp](https://api.checkface.ml/api/face/?value=example&dim=500&format=webp)

## /api/hashdata/

Get the latent
Use either `value`, `seed` or `guid` parameters to specify the latent

### Query Parameters

`value` *optional* **string** is any free text such as the hash of a file 

`seed` *optional* **int** corresponding to the seed used in the random number generator to generate latents

`guid` *optional* **guid** globally unique identifier returned by [/api/registerlatent](#post-apiregisterlatent)

### Returns

Json object with data about the latent used for the given value

```json
{
    "qlatent": [0.4221103225421117,-0.31882407908604915,0.36004541443342347,...],
    "seed": 78062172
}
```

### Example

[https://api.checkface.ml/api/hashdata/?value=example](https://api.checkface.ml/api/hashdata/?value=example)

## /api/morphframe/

Returns a single frame from a morph. Can be used to get the midpoint, 

Use `from_*` and `to_*` parameters to specify the start and end points of the morph.
You can mix and match values, seeds and guids.

### Query Parameters

`dim` *optional* **int** between 10-1024 is the size of the gif in px

`num_frames` *optional* **int** between 3-200 is the number of frames. Defaults to 50

`frame_num` **int** between 0 to (num_frames-1) specify the specific frame to return

`linear` **boolen** set true for a linear morph from start to end rather than trig based

`from_value`, `to_value` *optional* **string** is any free text such as the hash of a file 

`from_seed`, `to_seed` *optional* **int** corresponding to the seed used in the random number generator to generate latents

`from_guid`, `to_guid` *optional* **guid** globally unique identifier returned by [/api/registerlatent](#post-apiregisterlatent)

### Returns

An image

### Example

 [https://api.checkface.ml/api/morphframe/?from_value=example&to_value=yeet&frame_num=25&linear=true](https://api.checkface.ml/api/morphframe/?from_value=example&to_value=yeet&frame_num=25&linear=true)


## /api/gif/


Creates a gif morphing between two images.
Use either `from_*` and `to_*` parameters to specify the start and end points of the gif.
You can mix and match values, seeds and guids.

### Query Parameters

`dim` *optional* **int** between 10-1024 is the size of the gif in px

`fps` *optional* **int** between 1-100 is the frames per second

`num_frames` *optional* **int** between 3-200 is the number of frames in the gif

`from_value`, `to_value` *optional* **string** is any free text such as the hash of a file 

`from_seed`, `to_seed` *optional* **int** corresponding to the seed used in the random number generator to generate latents

`from_guid`, `to_guid` *optional* **guid** globally unique identifier returned by [/api/registerlatent](#post-apiregisterlatent)

### Returns

A gif

### Example

[https://api.checkface.ml/api/gif/?from_value=example&to_value=yeet&dim=400&fps=16&num_frames=80](https://api.checkface.ml/api/gif/?from_value=example&to_value=yeet&dim=400&fps=16&num_frames=80)

## /api/mp4/

Creates an mp4 morphing between two images.

Use either `from_*` and `to_*` parameters to specify the start and end points of the mp4.
You can mix and match values, seeds and guids.

Setting `embed_html=true` embeds the video in an HTML document, which automatically loops the video like a gif.
This is useful when trying out the api in a browser manually.

### Query Parameters

`dim` *optional* **int** between 10-1024 is the size of the mp4 in px

`fps` *optional* **int** between 1-100 is the frames per second

`num_frames` *optional* **int** between 3-200 is the number of frames in the mp4

`from_value`, `to_value` *optional* **string** is any free text such as the hash of a file 

`from_seed`, `to_seed` *optional* **int** corresponding to the seed used in the random number generator to generate latents

`from_guid`, `to_guid` *optional* **guid** globally unique identifier returned by [/api/registerlatent](#post-apiregisterlatent)

`embed_html` *optional* **boolean** determines whether the response is embedded in an HTML document and loops

### Returns

If `embed_html=true`, returns an html document, otherwise returns an mp4 video.

### Examples

 1. [https://api.checkface.ml/api/mp4/?from_value=example&to_value=yeet&dim=400&fps=16&num_frames=80](https://api.checkface.ml/api/mp4/?from_value=example&to_value=yeet&dim=400&fps=16&num_frames=80)
 2. [https://api.checkface.ml/api/mp4/?from_value=example&to_value=yeet&dim=400&fps=16&num_frames=80&embed_html=true](https://api.checkface.ml/api/mp4/?from_value=example&to_value=yeet&dim=400&fps=16&num_frames=80&embed_html=true)

## /api/linkpreview/

Creates an image with each endpoint and the midpoint.
Intented to be used as the link preview for a morph.

Use `from_*` and `to_*` parameters to specify the start and end points of the mp4.
You can mix and match values, seeds and guids.

### Query Parameters

`width` *optional* **int** between 100-1400 is the width of the image in px. Defaults to 1200 width and height of 628

`from_value`, `to_value` *optional* **string** is any free text such as the hash of a file 

`from_seed`, `to_seed` *optional* **int** corresponding to the seed used in the random number generator to generate latents

`from_guid`, `to_guid` *optional* **guid** globally unique identifier returned by [/api/registerlatent](#post-apiregisterlatent)

### Returns

An image

### Example

 [https://api.checkface.ml/api/linkpreview/?from_value=example&to_value=yeet](https://api.checkface.ml/api/linkpreview/?from_value=example&to_value=yeet)

## POST /api/registerlatent/

POST json to this endpoint to register a latent for later use.
It returns a guid which can be used later in guid parameters.

Use content type `application/json`.

### Body

The body should be a json object:

```json
{
    "latent": [0.12135718, -0.41143386,  0.77926035,...],
    "type": "test"
}
```

### Returns

Plaintext guid referencing the latent

For example

```
54fec40f-12c4-4333-951b-6bc1d2d074b9
```

### Example 

<details><summary>With curl (click here to expand)</summary>
<div markdown="1">

```bash
curl --location --request POST 'https://api.checkface.ml/api/registerlatent/' \
--header 'Content-Type: application/json' \
--data-raw '{
	"latent": [ 5, -0.33087015,  2.43077119, -0.25209213,  0.10960984,
        1.58248112, -0.9092324 , -0.59163666,  0.18760323, -0.32986996,
       -1.19276461, -0.20487651, -0.35882895,  0.6034716 , -1.66478853,
       -0.70017904,  1.15139101,  1.85733101, -1.51117956,  0.64484751,
       -0.98060789, -0.85685315, -0.87187918, -0.42250793,  0.99643983,
        0.71242127,  0.05914424, -0.36331088,  0.00328884, -0.10593044,
        0.79305332, -0.63157163, -0.00619491, -0.10106761, -0.05230815,
        0.24921766,  0.19766009,  1.33484857, -0.08687561,  1.56153229,
       -0.30585302, -0.47773142,  0.10073819,  0.35543847,  0.26961241,
        1.29196338,  1.13934298,  0.4944404 , -0.33633626, -0.10061435,
        1.41339802,  0.22125412, -1.31077313, -0.68956523, -0.57751323,
        1.15220477, -0.10716398,  2.26010677,  0.65661947,  0.12480683,
       -0.43570392,  0.97217931, -0.24071114, -0.82412345,  0.56813272,
        0.01275832,  1.18906073, -0.07359332, -2.85968797,  0.7893664 ,
       -1.87774088,  1.53875615,  1.82136474, -0.42703139, -1.16470191,
       -1.39707402,  0.87265462, -0.20211818, -0.59835993, -0.2434197 ,
        2.08851469,  0.34691933,  0.74572695,  0.77690759,  1.01842113,
        1.06135144, -0.71046645, -0.2151878 , -0.76076031, -0.71116323,
        1.14150774, -0.50175555, -0.07915136, -0.69282634, -0.59340277,
        0.78823794, -0.44542999, -0.48212019,  0.49355766,  0.50048733,
        0.79242262,  0.17076445, -1.75374086,  0.63029648,  0.49832921,
        1.01813761, -0.84646862,  2.52080763, -1.23238611,  0.72695326,
        0.04595522, -0.48713265,  0.81613236, -0.28143012, -2.33562182,
       -1.16727845,  0.45765807,  2.23796561, -1.4812592 , -0.01694532,
        1.45073354,  0.60687032, -0.37562084, -1.42192455, -1.7811513 ,
       -0.74790579, -0.36840953, -2.24911813, -1.69367504,  0.30364847,
       -0.40899234, -0.75483059, -0.40751917, -0.81262476,  0.92751621,
        1.63995407,  2.07361553,  0.70979786,  0.74715259,  1.46309548,
        1.73844881,  1.46520488,  1.21228341, -0.6346525 , -1.5996985 ,
        0.87715281, -0.09383245, -0.05567103, -0.88942073, -1.30095145,
        1.40216662,  0.46510099, -1.06503262,  0.39042061,  0.30560017,
        0.52184949,  2.23327081, -0.0347021 , -1.27962318,  0.03654264,
       -0.64635659,  0.54856784,  0.21054246,  0.34650175, -0.56705117,
        0.41367881, -0.51025606,  0.51725935, -0.30100513, -1.11840643,
        0.49852362, -0.70609387,  1.4438811 ,  0.44295626,  0.46770521,
        0.10134479, -0.05935198, -2.38669774,  1.22217056, -0.81391201,
        0.95626186, -0.63851056, -0.14312642, -0.22418983, -1.03849524,
       -0.17170905,  0.47634618, -0.41417827, -1.26408334, -0.57321556,
        0.24981732,  1.14720208,  0.83594396,  0.28740365, -0.9955963 ,
        0.90688947,  0.02421074, -0.23998173,  0.91011056,  0.61784475,
        0.49961804, -1.15154425, -0.6105164 , -1.70388541,  0.19443738,
        0.02824125,  0.93256051,  0.21204332, -0.36794457,  2.1114884 ,
       -1.02957349, -1.33628031, -0.61056736,  0.52469426, -0.34930813,
       -0.44073846, -1.1212876 ,  1.47284473, -0.62337224, -1.08070195,
       -0.12253009, -0.8077431 , -0.23255622,  1.33515034, -0.44645673,
       -0.04978868, -0.36854478, -0.19173957,  0.81967992,  0.53163372,
       -0.34161504, -0.93090048, -0.13421699,  0.83259361, -0.01735327,
       -0.12765822, -1.80791662,  0.99396898, -1.49112886, -1.28210748,
       -0.37570741,  0.03464388,  0.04507816, -0.76374689, -0.31313851,
       -0.60698954, -1.80955123, -0.25551774, -0.69379935,  0.41919776,
       -0.14520019,  0.9638013 ,  0.69622199,  0.89940546,  1.20837807,
        0.6932537 , -0.16636061,  1.35311311, -0.92862651, -0.03547249,
        0.85964595, -0.28749661,  0.71494995, -0.8034526 , -0.54048196,
        0.54617743,  0.71188926,  1.19715449, -0.07006703,  0.29822712,
        0.62619261,  0.46743206, -1.30262143, -0.57008965,  1.44295001,
       -1.24399513,  0.62888033, -0.42559213,  1.00320956, -0.77817761,
        0.04894463, -2.02640189, -0.04193635,  1.07454278, -1.5008594 ,
        1.18574443, -0.71508124, -0.05123853, -2.77458336,  1.07862813,
       -0.87568592, -0.53810932, -1.2782157 , -0.99276945,  1.14342789,
       -0.5090726 ,  0.89500094, -0.17620337,  0.34608347, -0.50631013,
        0.42716402,  2.58856959,  0.65289301,  0.50583979, -0.47595083,
        1.01090874,  1.35920097, -1.70208997, -1.38033223,  2.10177668,
        0.42589917,  0.12920023,  0.56296251,  1.09676472,  0.80081885,
       -0.22308327,  2.06367066,  0.0126235 , -0.8747738 , -0.55707938,
       -0.13230195, -0.37922499, -0.18779371,  0.31546615, -3.28391545,
       -0.77869325,  0.95034471,  0.5630013 , -0.68065407, -0.62450339,
        1.14049594, -0.24772894, -0.53020527,  1.8557144 , -0.36987213,
        0.68424682, -0.0456703 ,  0.05078665, -0.94722556, -0.82698742,
        1.25807361, -1.13889026,  0.27736012, -1.19444596, -0.24043683,
       -0.03720827, -1.6296784 ,  1.13486338, -0.18379943,  1.21473773,
       -0.93427859,  0.91186241,  2.3342401 ,  0.21653196, -0.64706848,
        0.47870605,  0.14082715, -0.2099986 , -0.12050664, -0.57882578,
        0.42386759, -0.38733136, -0.85686815,  0.81531389, -0.16581602,
        2.64535345, -0.24946988, -0.71733789, -0.54949733,  0.37108695,
       -0.69734581, -1.26330116,  1.63921233, -1.24014464,  1.51364577,
        0.14105657, -1.06209796,  1.6663804 , -0.2034536 , -1.00754753,
        0.06540956,  1.28644574,  0.68374332,  0.8262448 ,  1.75433632,
        0.21456398,  0.37581479, -0.22598417, -1.45469387, -0.14453466,
        1.61697881, -1.73105363,  1.34394613,  0.26153957, -0.91051935,
        0.06546949,  1.77632213, -0.57313319,  0.79059361,  1.13151397,
       -0.897094  ,  0.63271186,  0.53515515, -0.47415241,  0.68498591,
       -0.36119419, -0.57742993, -1.2347295 ,  0.38547989, -0.42918999,
       -0.55892627, -1.14899998, -1.36515578, -0.78923902,  0.72995982,
       -0.81388187,  1.4448595 ,  0.40825946,  0.15806514, -1.20324067,
        1.95358868, -1.4406335 ,  0.53407511,  1.69432832, -0.19894722,
       -0.68352568, -0.01899812,  0.9156626 ,  1.35870723,  0.60443768,
       -1.06941562, -0.6741898 ,  0.20340805, -1.27616516, -0.24030333,
        2.24095357, -1.05746192,  1.16055901, -0.93298444, -0.34072389,
       -0.07013113, -1.50552315, -0.10507983,  1.29682083,  0.7171925 ,
        0.69777111, -0.80449784, -0.14505178,  0.2023229 ,  0.67869955,
        1.34251188, -0.99933073,  1.69954809, -0.28621623, -0.25163697,
       -1.20844686, -0.06779508, -0.22818598, -1.23450433, -0.29138373,
        0.12135718, -0.41143386,  0.77926035, -1.02468459,  0.88988217,
       -0.18598247,  0.37226978, -1.84518514,  0.12914587, -0.06190023,
        0.9357079 , -1.17990317,  1.36404151, -1.08263117,  1.31669419,
        0.57819563, -0.7544614 ,  2.16976159, -1.19562434, -0.17197421,
        0.20706706,  0.52178374,  0.22638929,  0.79913028,  0.45924581,
        0.03269967, -0.92956292, -0.345037  ,  0.90247952, -1.16649931,
        0.11099181, -2.04839658, -0.69561095, -1.62316059,  1.24454078,
       -1.82274919, -0.2396064 ,  0.72844306,  0.60888427,  0.77318471,
        1.06235383,  0.47350502,  0.83459787, -0.05414128, -0.02563969,
       -1.76040064,  0.16870521,  1.26727682, -0.7479485 , -1.16974715,
        0.09123447,  1.13441899],
	"type": "test"
}'
```

</div>
</details>

## POST /api/encodeimage/

POST an image to this endpoint to encode it as a latent.
It returns a guid which can be used later in guid parameters.

Use `multipart/form-data` content type.
Make sure the file is a valid image file and is not too big.

### Form Data

`tryalign` **boolen** set true to try aligning the face

`usrimg` **file**  an image file

### Returns

Json object containing the guid and whether or not it aligned the face

```
{"did_align":false,"guid":"54fec40f-12c4-4333-951b-6bc1d2d074b9"}
```

### Example

[https://api.checkface.ml/api/encodeimage/](https://api.checkface.ml/api/encodeimage/)

<script>
(async function() {
    // Get a hash of the source for this page to make a checkface for this page
    let resp = await fetch(window.location.href)
    const digest = await crypto.subtle.digest("SHA-256", await resp.arrayBuffer());
    const hashArray = Array.from(new Uint8Array(digest));                     // convert buffer to byte array
    const hashB64 = btoa(String.fromCharCode.apply(null, new Uint8Array(hashArray))); //encode bytes as base64
    let pageVersion = encodeURIComponent(hashB64.substr(0, 10));
    let imgSrc = `https://api.checkface.ml/api/face/?value=${pageVersion}&dim=50`
    let linkHref = `/?value=${pageVersion}`
    document.getElementById("page-version-checkface-img").src = imgSrc;
    document.getElementById("page-version-checkface-link").href = linkHref;
})();
</script>