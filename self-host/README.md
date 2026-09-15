# CheckFace API and original FaceMorpher UI — local candidate

This source candidate reuses the deployed Flask API logic and the existing
F# / Fable / Elmish / React UI. Modern CPU PyTorch replaces the obsolete NVIDIA
container runtime; Mongo retains saved latents and GUIDs, and FFmpeg produces the
existing animation formats. No HF service, GPU driver or Triton access is needed.
No image, model bundle or release artifact is published.

## Try it from a fresh clone

Install Docker with Compose v2; allow about 8 GiB RAM and 12 GiB free disk for
build tools, runtime and model assets. Review the licensing links below first.

```sh
git clone --branch candidate/self-host-api https://github.com/check-face/checkface.git
cd checkface/self-host
docker compose build
docker compose run --rm setup --accept-research-license
docker compose up -d --wait --wait-timeout 300 api
```

Open **http://localhost:8080** for the original UI. It uses this local API;
photos and latents stay in this Compose project. Setup downloads approximately
1.7 GB of original model assets from their authors and checks fixed byte counts
and SHA-256 hashes before loading anything. Downloads are atomic and repeat
setup reuses verified files. No account or API key is needed. The generator is
converted in memory by NVIDIA's pinned adapter; TensorFlow is not installed.
The API admits eight generation requests with sixteen HTTP workers, reserving
capacity for health/status; excess generation receives 503 with `Retry-After`.
CPU generation is slower than the deployed GPU, especially long animations.

```sh
curl 'http://localhost:8080/api/face/?seed=0&dim=512' --output face.jpg
curl 'http://localhost:8080/api/hashdata/?value=hello'
curl 'http://localhost:8080/api/mp4/?from_seed=0&to_seed=1&dim=256&num_frames=4' --output morph.mp4
curl -F usrimg=@face.jpg -F tryalign=true 'http://localhost:8080/api/encodeimage/'
docker compose logs api
docker compose down
```

`down` preserves models, generated media, latent records and upload-deduplication
records. **`down --volumes` deletes all of those local project volumes**, including
saved GUIDs. The API binds only to loopback; Mongo has no published port. Change
the loopback port mapping if 8080 is occupied. This candidate is for local testing;
public deployment and production cutover are separate work.

## Retained API behavior

Existing parameter details are in [the original API documentation](../docs/api.md).
The implementation and [route inventory](legacy-route-inventory.json) preserve:

| Routes | Retained behavior |
|---|---|
| `/api/face/`, `/api/<text>` | Text, numeric seed or saved GUID; JPEG/WebP; dimension defaults and bounds; original text hash and seed mapping. |
| `/api/hashdata/`, `/api/registerlatent/` | Original JSON latent representation and UUID response; Z and W+; original blend/style parameters. |
| `/api/morphframe/` | Original linear slider and sinusoidal looping schedules; mixed Z/W+ endpoints. |
| `/api/gif/`, `/api/webp/`, `/api/mp4/` | Original looping frames, frame count, dimensions, FPS/bitrate and MP4 embed option; cached outputs. |
| `/api/linkpreview/` | Original composite preview generation with the deployed static branding assets. |
| `/api/encodeimage/` GET/POST | Existing upload form and multipart `usrimg`/`tryalign`; original e4e and dlib preprocessing; `{guid,did_align}` response and deduplication. |
| `/api/queue/`, `/status/` | Existing queue/status responses. |
| `/oembed.json` | Original UI photo/embed metadata with local URLs. |
| `/`, static UI paths | Original FaceMorpher UI configured for same-origin API calls. |
| `/healthz` | Additional readiness check for local Compose. |

The separate early `/v1` prototype is no longer the served API. New API features,
figure-eight/custom modes and GPU packaging are not prerequisites for this
compatibility candidate. Existing frontend production defaults are preserved;
self-host settings are a small overlay on a pinned original source revision.

Practical differences are explicit: CPU timings, bounded request admission and
clearer invalid-input errors; modern PyTorch/Pillow/FFmpeg can change numeric
rounding or compressed bytes. This is not a claim of byte-identical historic JPEGs
or of production encoder parity. The generator uses the original checkpoint,
constant noise, full 1024px synthesis and truncation psi 0.7/cutoff 8.

A fresh installation has **its own empty database and cache**. Existing public
GUIDs require an authorized restore of their records; URLs alone do not recover
private latents. The candidate retains Mongo database `test`, collections
`latents` and `encodedimages`, and the original cache directory layout under
`/app/checkfacedata`. Restore a copy into this project's volumes and verify it
independently before migration. Do not mount production volumes or assume that
copying historical Mongo data files across major versions is safe; use a logical
Mongo backup/restore with supported tools. Preservation and live cutover remain
separate, explicitly approved operations.

## Validation

The candidate CI builds its own image, acquires the real models and exercises
actual HTTP inference. It publishes no image, model, installer or release.
Compatibility checks cover retained routes, actual media, encoding and persistence;
mock-only unit tests do not establish end-to-end completion. Run the same checks locally:

```sh
docker compose exec -T api python smoke_frontend.py
docker compose exec -T api python test_legacy_contract.py
docker compose exec -T api python test_encoder_assets.py
docker compose exec -T api python test_encoder.py
docker compose exec -T api python smoke_legacy.py
docker compose exec -T api python smoke_encoder_http.py
docker compose restart api
docker compose up -d --wait --wait-timeout 300 api
docker compose exec -T api python smoke_legacy.py
docker compose exec -T api python smoke_encoder_http.py
```

The repeated HTTP checks verify existing GUIDs, encoded-upload records and media
survive restart. Long-morph unit checks cover both 199 and 200 frames with bounded
live image buffers. See [candidate CI runs](https://github.com/check-face/checkface/actions/workflows/self-host-build.yml)
for the tested commit and complete Linux build/inference logs.

Python/Torch/direct dependencies and source revisions are pinned. Base-image
digests, complete dependency hashes, distribution SBOM and broader platform
qualification remain release work. Linux arm64 and amd64 are the target test
platforms; other architectures and GPU acceleration are not claimed.

## Model and source terms

Review each component's terms for your intended use before accepting setup:

- [Original NVIDIA StyleGAN2 terms](https://github.com/NVlabs/stylegan2/blob/master/LICENSE.txt) and [pinned conversion code terms](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/d72cc7d041b42ec8e806021a205ed9349f87c6a4/LICENSE.txt) restrict use to research/evaluation and require retained notices. The latter stays at `/opt/stylegan/LICENSE.txt`.
- [e4e source and checkpoint provenance](https://github.com/omertov/encoder4editing) and the vendored [MIT source license](e4e/LICENSE); source licensing alone does not replace model/data terms.
- [dlib landmark-model notice](https://dlib.net/face_landmark_detection.py.html) identifies the iBUG training dataset's noncommercial restriction.
- CheckFace's [CC BY-NC 4.0 license](../LICENSE.txt), the original frontend's source provenance, and third-party dependency notices also apply. No new license for the frontend is invented by this overlay.

Local building is not an exemption from usage restrictions. The setup flag records
acknowledgement; it grants no rights. General public/commercial operation or
redistribution needs its own licensing review. This candidate shares source only.

## CPU allocation and retained originals (15 September follow-up)

`CHECKFACE_CPU_THREADS=auto` selects at most eight PyTorch model threads, bounded
by CPU count, process affinity and visible cgroup v1/v2 CPU quotas (including
ancestors). Fractional quotas round down, with a minimum of one thread. Override
with an integer 1–64; overrides are capped to detected allocation. Invalid values
fail startup. For example, `CHECKFACE_CPU_THREADS=4 docker compose up -d api`.
`/healthz` and startup logs report requested/effective settings. Hidden host quotas
cannot be discovered; set a smaller override when necessary. This is a bounded
starting policy, not a benchmarked optimum. Small/large allocation timing and
health responsiveness under real inference still require qualification.

These are model intra-op threads, also used by the encoder in this process.
HTTP concurrency remains sixteen workers/eight generation admissions; changing
model threads does not create extra inference jobs. FFmpeg retains its separate
two-thread limit. CPU is the only qualified provider in this image.

The legacy generator adapter now writes private lossless 1024×1024 RGB PNGs to
`cache:/app/checkfacedata/originals-v1` **before resizing**. Every synthesis route
uses this adapter, including queued faces and morph frames. Different sizes and
formats reuse these raw pixels. Historic JPEG/WebP/media paths are untouched and
continue serving existing artifacts; they never count as original PNGs. The
separate early `/v1` prototype does not use this store.

Each original embeds a canonical identity and raw-pixel SHA-256. Identity includes
the verified loaded checkpoint hash, exact little-endian float32 resolved W+
bytes/shape, checkpoint-constant noise, output-pipeline version, PyTorch version,
CPU provider and effective thread count. W+ identity is **after** mapping,
truncation and any photo preprocessing, so changed image-affecting inputs change
the key. It does not retain uploaded photo bytes or reconstruct their preprocessing
history. These originals can include photo-derived faces: this is private local
storage, **not the synthetic-only public preservation archive**. Bump the output
pipeline version whenever pixel conversion/synthesis semantics change.

A process/file lock deduplicates synthesis; each PNG atomically carries pixels and
provenance, with fsync before promotion. Interrupted staging and corrupt entries
are retried safely. `CHECKFACE_ORIGINAL_CACHE_BYTES` defaults to 10 GiB and limits
this namespace only; it does not limit historic derivatives/models. At capacity
or storage failure the generated result still returns, a warning/counter records
failed persistence, and existing originals are retained. There is no automatic
eviction. To reclaim space, stop the local API and explicitly remove selected
original PNGs from this namespace; later requests can regenerate them. Increasing
the quota requires recreating the API container. Cache bookkeeping currently
scans PNG sizes on writes; very large collections may need an indexed quota ledger.

`/healthz.original_cache` reports process-local hits, generated originals and
write failures. A restart retains pixels, but this server still initializes the
model eagerly and recomputes mapping before a W+ lookup; avoiding that startup
work is not claimed. Runtime/model/thread changes retain old files under separate
keys. Original-cache identity does not rewrite the historical derivative namespace.

Lightweight checks: `python test_cpu_threads.py`, `python test_original_cache.py`.
Inside the image also run `python test_adapter_cache.py` (real torch tensors,
fixture generator). Real-model HTTP evidence requires `smoke_original_cache.py`
before and after an API restart with its state file retained outside `/tmp` in the
recreated container, alongside the existing full-route/encoder smoke suite. The
new adapter integration is source work until those real-model checks pass.
