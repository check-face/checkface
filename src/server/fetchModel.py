# load stylegan2 model
import dnnlib
#'gdrive:networks/stylegan2-ffhq-config-f.pkl'
url = 'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-f.pkl'
dnnlib.util.open_url(url, cache_dir='.stylegan2-cache')