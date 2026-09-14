// Pass a generated Config.fs.js path (from dotnet fable); no browser/server needed.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
const source = pathToFileURL(process.argv[2]);
const supportsTrial = readFileSync(process.argv[2], 'utf8').includes('FACEMORPH_TRIAL');
globalThis.window = { location: { origin: 'http://127.0.0.1:8080', hostname: '127.0.0.1', pathname: '/classic' } };
const cases = [
  {self:'1', trial:'1', review:'1', api:'', canonical:window.location.origin, classic:false},
  {self:'0', trial:'0', review:'0', api:'https://api.facemorph.me', canonical:'https://facemorph.me', classic:false},
  {self:'0', trial:'1', review:'1', api:supportsTrial ? '/trial' : 'https://api.facemorph.me', canonical:supportsTrial ? window.location.origin : 'https://facemorph.me', classic:true},
];
for (const [i, test] of cases.entries()) {
  Object.assign(process.env, { FACEMORPH_SELF_HOST:test.self, FACEMORPH_TRIAL:test.trial, FACEMORPH_REVIEW:test.review });
  const config = await import(source.href + '?test=' + i);
  assert.equal(config.apiAddr, test.api);
  assert.equal(config.encodeApiAddr, test.api + '/api/encodeimage/');
  assert.equal(config.canonicalBaseUrl, test.canonical);
  assert.equal(config.oEmbedApiEndpoint, test.canonical + '/oembed.json');
  assert.equal(config.isClassicReview, supportsTrial ? test.classic : undefined);
}
console.log('Passed 15 emitted-config checks: self-host same-origin override and prior hosted/trial defaults.');
