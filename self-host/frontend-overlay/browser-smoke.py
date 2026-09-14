# Run with: CHECKFACE_TEST_URL=http://127.0.0.1:8080 browser-harness < this-file
# Uses the browser-harness pre-imported helpers; creates its own tab.
import json
import os
from urllib.parse import urlsplit

url = os.environ.get('CHECKFACE_TEST_URL', 'http://127.0.0.1:8080')
assert urlsplit(url).hostname in ('127.0.0.1', 'localhost'), 'This smoke is for the local candidate only.'
new_tab(url)
wait_for_load()
print(page_info())
# Read the actual mounted UI and browser resource records, not static HTML alone.
report = json.loads(cdp('Runtime.evaluate', expression='''JSON.stringify({
  origin:location.origin,
  mounted:!!document.querySelector('#elmish-app')?.children.length,
  controls:[...document.querySelectorAll('input')].map(e=>({id:e.id,type:e.type,value:e.value})),
  images:[...document.images].map(e=>({src:e.currentSrc||e.src,loaded:e.complete&&e.naturalWidth>0})),
  links:[...document.querySelectorAll('link[rel="canonical"]')].map(e=>e.href),
  resources:performance.getEntriesByType('resource').map(e=>e.name),
  frames:[...document.querySelectorAll('iframe')].map(e=>e.src)
})''', returnByValue=True)['result']['value'])
assert report['mounted'], 'Original Elmish app did not mount.'
assert len(report['controls']) >= 2, 'Expected original endpoint input controls.'
requests = report['resources'] + [entry['src'] for entry in report['images']] + report['frames']
blocked = [request for request in requests if any(host in request for host in
    ('api.facemorph.me', 'names.facemorph.me', 'huggingface.co', '.hf.space', 'googletagmanager.com'))]
assert not blocked, ('Unexpected production/trial request', blocked)
for image in report['images']:
    if '/api/' in image['src']:
        assert image['src'].startswith(report['origin'] + '/api/'), image
assert all(link.startswith(report['origin']) for link in report['links'])
print(json.dumps(report, indent=2))
# Root agent can continue coordinate-based clicks from these real AX nodes.
for node in cdp('Accessibility.getFullAXTree')['nodes']:
    role = node.get('role', {}).get('value')
    name = node.get('name', {}).get('value', '')
    if role in ('button','textbox','slider','link'):
        print(role, repr(name), node.get('backendDOMNodeId'))
