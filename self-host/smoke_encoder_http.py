"""Real multipart upload -> stored W+ GUID -> reconstruction/morph/restart proof."""
import hashlib
import io
import json
import os
from pathlib import Path
import uuid
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

BASE = os.getenv('CHECKFACE_TEST_URL', 'http://127.0.0.1:8080')
DATA = Path(os.getenv('CHECKFACE_DATA_DIR', '/app/checkfacedata'))
FIXTURES = Path(__file__).with_name('fixtures')

def get(path, params):
    with urlopen(BASE+path+'?'+urlencode(params), timeout=1200) as response:
        return response.read()

def upload(image, align=None):
    boundary='checkface-'+uuid.uuid4().hex
    parts=[]
    if align is not None:
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="tryalign"\r\n\r\n{str(align).lower()}\r\n'.encode())
    parts.extend([f'--{boundary}\r\nContent-Disposition: form-data; name="usrimg"; filename="synthetic.png"\r\nContent-Type: image/png\r\n\r\n'.encode(), image, f'\r\n--{boundary}--\r\n'.encode()])
    request=Request(BASE+'/api/encodeimage/',data=b''.join(parts),headers={'Content-Type':f'multipart/form-data; boundary={boundary}'})
    with urlopen(request,timeout=1200) as response:
        result=json.load(response)
    uuid.UUID(result['guid'])
    assert type(result['did_align']) is bool
    return result

def latent(guid):
    payload=json.loads(get('/api/hashdata/',{'guid':guid}))
    assert 'dlatent' in payload and 'qlatent' not in payload
    result=np.asarray(payload['dlatent'],dtype=np.float32)
    assert result.shape==(18,512) and np.isfinite(result).all()
    return result

def image(path,params):
    raw=get(path,params)
    with Image.open(io.BytesIO(raw)) as result:
        assert result.format=='JPEG' and result.size==(64,64)
        result.load()
    return raw

def main():
    source=(FIXTURES/'seed-0.png').read_bytes()
    receipt=DATA/'self-host-encoder-validation.json'
    previous=json.loads(receipt.read_text()) if receipt.exists() else None
    aligned=upload(source,True)
    assert aligned['did_align'] is True
    # New multipart boundaries must not change the image-bytes + tryalign cache key.
    assert upload(source,True)==aligned, 'Repeated upload allocated a new GUID'
    actual=latent(aligned['guid'])
    reference=np.load(FIXTURES/'seed-0-encoded-w-plus.npy',allow_pickle=False)
    error=float(np.abs(actual-reference).max())
    assert error<=.0001, f'Uploaded W+ differs from prior CPU encoder: {error}'
    unaligned=upload(source,False)
    assert unaligned['did_align'] is False and unaligned['guid']!=aligned['guid']
    assert upload(source)==unaligned, 'Omitted tryalign must default to false'
    latent(unaligned['guid'])
    blank=io.BytesIO(); Image.new('RGB',(256,256),(128,128,128)).save(blank,format='PNG')
    noface=upload(blank.getvalue(),True)
    direct=upload(blank.getvalue(),False)
    assert noface['did_align'] is False and direct['did_align'] is False
    np.testing.assert_array_equal(latent(noface['guid']),latent(direct['guid']))

    reconstruction=image('/api/face/',{'guid':aligned['guid'],'dim':64})
    morph={'from_guid':aligned['guid'],'to_seed':1,'linear':'true','num_frames':3,'dim':64}
    assert image('/api/morphframe/',{**morph,'frame_num':0})==reconstruction
    middle=image('/api/morphframe/',{**morph,'frame_num':1})
    assert middle!=reconstruction
    last=image('/api/morphframe/',{**morph,'frame_num':2})
    assert last==image('/api/face/',{'seed':1,'dim':64})
    image('/api/face/',{'num_multi':2,'guid0':aligned['guid'],'seed1':1,'amount0':.5,'amount1':.5,'dim':64})
    current={'guid':aligned['guid'],'did_align':True,'source_sha256':hashlib.sha256(source).hexdigest(),
             'w_plus_sha256':hashlib.sha256(actual.astype('<f4').tobytes()).hexdigest(),
             'reconstruction_sha256':hashlib.sha256(reconstruction).hexdigest()}
    if previous is not None:
        assert current==previous, 'Stored uploaded GUID/latent/cache did not survive previous run/restart'
    DATA.mkdir(parents=True,exist_ok=True)
    receipt.write_text(json.dumps(current,indent=2)+'\n')
    print(json.dumps({'multipart_face_alignment':True,'max_encoder_reference_error':error,'repeat_guid_dedup':True,
                     'unaligned_default':True,'no_face_fallback':True,'w_plus_retrieval':True,'reconstruction_and_mixed_morph':True,
                     'previous_receipt_survived':previous is not None,'receipt':str(receipt)}))

if __name__=='__main__': main()
