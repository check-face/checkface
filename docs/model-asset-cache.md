# Persistent model and runtime asset cache

Required product contract, 15 September 2026. Implementation and cross-platform verification are pending. Applies to all released model versions and browser/native CPU/GPU routes, plus self-host deployment.

## Goal

Download each required immutable asset once per available storage context, retain it indefinitely by default, and reuse it across sessions, app upgrades, model bundles and provider switches wherever the bytes are identical. Return visits must not transfer model bytes already stored intact. Download only the dependencies needed for the selected workflow; do not eagerly download every version or CPU/GPU variant.

“Basically forever” means no application-imposed expiry or routine purge for verified model bytes. Browsers, operating systems and users can still clear or deny storage. CDN/HTTP freshness does not guarantee physical retention at every edge/browser. Keep an independently preserved origin copy of every supported released asset and recover missing local data without changing project identity.

## Content identity and all model versions

- Give every model graph, weight shard, mapping/synthesis asset, e4e model, landmark/alignment asset, noise/reference fixture, runtime JS/WASM/native component and codec asset a SHA-256, exact size and immutable canonical URL where distribution is permitted.
- A versioned bundle manifest maps model/version/provider roles to hashes and records compatibility and provenance. Identical bytes retain the same hash/URL across releases. UI-only changes must not change model URLs. Different bytes always get new keys; never overwrite an immutable asset.
- CPU/GPU graphs can differ despite sharing a checkpoint. Reuse byte-identical components only; do not infer identical assets from semantic equivalence. Do not require re-downloading an unchanged whole bundle because one component changed.
- Avoid app-build cache-busters, random query parameters and expiring signatures in asset identity. If acquisition requires temporary credentials, keep the stable content hash as the local key and secrets out of logs/metadata.
- Revalidate small mutable discovery/release metadata. Never cache a mutable `latest` pointer permanently. Keep pinned immutable manifests and dependencies available for old projects and rollback.

## Cache layers

| Layer | Contract |
|---|---|
| Durable origin and independent backup | Retain released assets/manifests for permitted distribution, old projects and rollback. CDN misses retrieve existing objects, never regenerate them or depend on Triton. |
| CDN and browser HTTP cache | Stable distributable blob URLs use `Cache-Control: public, max-age=31536000, immutable`. Verify actual served headers and CORS. HTTP cache alone is insufficient for reliable offline retention. |
| Application-managed browser storage | Explicitly persist verified bytes using a qualified Cache Storage or OPFS/IndexedDB implementation. Read locally before networking. Request persistent storage where supported, record whether granted and handle quota/refusal. Use one authoritative blob store per context, rather than duplicate every large file into several stores. |
| Active session | Reuse compatible initialized models when useful, but release inactive GPU buffers/sessions to bound RAM. Persistent disk caching does not mean every model stays loaded. |
| Native app | Stable per-user content-addressed storage outside replaceable installer/app-version directories. App updates and provider changes reuse retained assets. |
| Self-host Docker | Persist downloaded/converted models in volumes outside disposable containers. Rebuilds reuse verified files; ordinary deployment does not wipe model volumes. |

Storage is scoped by origin/profile and may be partitioned in embedded contexts. Do not promise automatic sharing between trial/production/names origins, different browsers, or browser/native apps. Keep a stable generation origin and avoid unnecessary origin changes. Names browsing needs cached images, not model downloads. An explicit verified model import can be considered separately.

Browser persistence remains quota-bound and cannot defeat user deletion; private browsing can have different retention. See [storage quotas and eviction](https://developer.mozilla.org/en-US/docs/Web/API/Storage_API/Storage_quotas_and_eviction_criteria). Header semantics: [HTTP caching](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/Caching). Checked 15 September 2026.

## Download, retention and recovery

1. Resolve output-image caches first. A face-cache hit must not download or initialize models just for viewing.
2. Resolve exact dependencies for the requested workflow/provider, check retained hashes and download only missing assets. Bound concurrency and deduplicate in-flight work across jobs/tabs where supported.
3. Write into incomplete staging. Stream/checkpoint verified shards or supported ranges without holding duplicate full files in RAM. Resume only against the same immutable identity; verify final size/hash before atomic promotion. Partial files are never valid hits. Recover only missing/corrupt pieces where independently verifiable.
4. Track transfer, verification, persistence, initialization and inference separately. Downloaded does not imply saved or ready. Retaining bytes avoids network transfers, not necessarily parsing, compilation or GPU upload.
5. App/service-worker updates must not clear the model store. No routine age-based deletion. Retain previously downloaded model versions by default; preserve project dependencies and blob identities through migrations. Do not use blanket “delete old caches” activation logic for model storage.
6. Provide storage usage and explicit removal of unused versions/downloads, explaining future download needs. Under quota pressure, reclaim incomplete/temporary/derived data first. Do not silently discard pinned models, generated originals or user work. If space remains insufficient, offer storage management or nonpersistent use where feasible; prevent download/retry loops.
7. Admission and compiled-runtime validity are separate from model-byte validity. Browser/driver/runtime updates may require fresh canaries or compilation, not re-downloading unchanged models. Known-bad/revoked runtimes must not execute simply because they are cached; record compatibility/revocation and offline limitations.
8. Apply these rules to old/new released versions, CPU/GPU paths, photo processing and codecs. Lazy acquisition avoids downloading unused GPU/e4e assets on a CPU-only seed workflow, while provider fallback reuses all identical dependencies.

User-facing copy: “Model files are saved on this device for future visits. If your browser clears its storage, they may need to download again.” Only claim successful persistence after it is confirmed.

## Acceptance across the support matrix

- After complete initial setup, repeat generation, reload and browser/app restart transfer **zero bytes for already-retained immutable assets**. Small metadata checks may still occur.
- A new uncached synthetic request works offline on an admitted route when all required app/runtime/model assets are present. Showing a previously cached image is not proof of offline inference.
- UI-only upgrades transfer no unchanged models. An A→B model upgrade fetches only new blobs; reopening A reuses retained A assets. CPU/GPU switches reuse common hashes and acquire only different dependencies.
- Interrupted downloads resume safely; corrupt/partial files recover. Concurrent jobs/tabs avoid redundant transfers. Measure peak download/verification memory on constrained devices.
- Verify persistence granted/denied, private mode, eviction/clearing, quota errors, service-worker migration, native updates and Docker rebuilds. Test exact asset hashes and transferred bytes, not just load times.
- Record cold transfer, warm local read and initialization separately per actual browser/native OS/architecture/provider. Research cold runs use isolated test storage or run-scoped bypass; never clear the user's retained models to benchmark.

No single desktop or Simulator result establishes retention/offline behavior across the entire matrix. This contract complements the generated lossless 1024px original cache; it does not replace it.
