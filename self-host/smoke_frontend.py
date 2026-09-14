"""Verify the built original UI is actually served with local assets and sharing."""
import json
import os
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit, urlencode
from urllib.request import urlopen

BASE = os.getenv('CHECKFACE_TEST_URL', 'http://127.0.0.1:8080')

class Assets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []
        self.mounted_root = False
    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        self.mounted_root |= attributes.get('id') == 'elmish-app'
        if tag == 'script' and attributes.get('src'):
            self.urls.append(attributes['src'])
        if tag == 'link' and attributes.get('rel') == 'stylesheet':
            self.urls.append(attributes['href'])

with urlopen(BASE, timeout=10) as response:
    document = response.read()
    assert response.headers.get_content_type() == 'text/html'
parser = Assets()
parser.feed(document.decode())
assert parser.mounted_root and len(parser.urls) >= 3, 'Missing original compiled UI'
for asset in parser.urls:
    url = urljoin(BASE, asset)
    assert urlsplit(url).netloc == urlsplit(BASE).netloc, ('Nonlocal runtime asset', url)
    with urlopen(url, timeout=10) as response:
        assert response.status == 200 and len(response.read()) > 100
assert b'googletagmanager.com' not in document
for route in ('/retirement', '/explain'):
    assert urlopen(BASE + route, timeout=10).read() == document
with urlopen(BASE + '/oembed.json?' + urlencode({'url': BASE + '/?from_seed=0&to_seed=1'}), timeout=10) as response:
    embed = json.load(response)
    assert embed['url'].startswith(BASE + '/api/')
print(json.dumps({'original_ui_static_assets': len(parser.urls), 'spa_paths': True, 'local_oembed': True}))
