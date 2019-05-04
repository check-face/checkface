import sys
sys.path.append('/app/dnnlib')
import dnnlib

url = 'https://drive.google.com/uc?id=1-O8VHNOpBNHnQyn0yz_pK3PHoc3CboC3'

with dnnlib.util.open_url(url, cache_dir='cache') as f:
    print("Downloaded model to cache")