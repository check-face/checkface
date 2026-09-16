# Public synthetic face archive

Status: agreed scope and execution contract, 15 September 2026. Export, classification, hosting, shared access and restore verification are not yet complete. This document must become the as-built runbook as each step is performed; do not present planned commands or resources as deployed.

## Scope

Preserve **all existing cached seed/hash-generated images from Triton**, including active caches and the separate oldSeeds tree. A small names/common-face collection may be the first delivery batch, but is not the completion gate. Include all existing image variants and record source/model-era differences without overwriting one with another. Preserve exact stored bytes; regeneration is a separate operation.

The verified synthetic archive is deliberately public. Anyone may enumerate it, download it in full and mirror it. Publish a machine-readable manifest and resumable bulk-download instructions. Opaque names, secret indexes and anti-enumeration services are not required.

Exclude e4e/photo-derived images, upload-derived latent outputs, mixed results containing any uploaded latent, private records and unresolved provenance. A hexadecimal filename is not sufficient proof: upload caches also use hashes. A GUID may identify arbitrary registered latents; it does not establish a synthetic seed origin. Quarantine uncertain artifacts privately with an exclusion reason and investigate them. Exclusion from this public archive does not authorize deletion of private historic data or cancel its separate preservation requirements.

All names-site entries must be covered. Preserve exact name case/order/face mapping and existing visual assets; missing names-cache entries require an explicit completion task. Do not invoke the production API to fill misses as part of read-only inventory. Names-page appearance and behavior remain unchanged.

## Execution and evidence

1. **Inventory read-only.** Enumerate active cache roots and `/mnt/unit/checkfacedata-oldSeeds`, including image/frame files in media directories. Record file/byte counts and naming rules. Inspect source and, where needed, read-only metadata relationships to distinguish seed/text-derived from uploaded/mixed data. Do not scan unrelated host data. No deploys, restarts, production generation or source-file edits.
2. **Classify and reconcile.** Build a private provenance ledger with source root/path, request identity, dimensions/format, model-era evidence, eligibility rule, SHA-256 and exclusion reason. Count all files as included, excluded, unresolved, unreadable or changed during copying. No unaccounted files. Do not classify an entire mixed directory as public.
3. **Copy and verify.** Perform resumable reads into staging outside Triton; retain independent checksummed backups. Detect incomplete or concurrently changing files, retry stable reads and document gaps. Deduplicate identical bytes while retaining aliases/provenance. Capture a final delta before cutover. Do not delete source data.
4. **Size the complete public release.** Measure unique object bytes, indexes, optional download bundles, retained versions and expected requests. Include other account usage in free-tier headroom. If the full archive cannot fit, report the measured gap and choose an explicit hosting/budget alternative; do not silently omit long-tail seeds to meet a quota.
5. **Build a portable publication.** Use immutable versioned objects and a sanitized public manifest containing lookup keys, artifact checksums, byte sizes, dimensions, format and relative download paths. Include numeric seeds and hash identities needed for lookup, plus the already-public names catalogue; omit raw private text, upload metadata, internal paths and credentials. A public hash manifest reveals membership by design. Shard indexes for fast client lookup without downloading every image; a bulk downloader traverses the complete manifest. If bundles are added, count their duplicate storage cost.
6. **Upload and verify.** Upload only the approved manifest's artifacts. Check object integrity, cache headers, cross-origin image fetching, missing keys and bulk download verification. Public clients get read-only access and cannot upload, delete or list private staging. Avoid a database or Worker solely to hide the public index. A thin compatibility resolver is justified only by existing URL/query contracts.
7. **Independent maintenance rehearsal.** Chris and Oliver each use their own account to inspect the release and perform a documented staging update/verification/rollback. At least one restores the archive from an independent backup without the other's machine or login. Record results before calling the archive maintainable.

Completion means all eligible cached images are accounted for and preserved, all public release objects match the published manifest, every names entry resolves, bulk downloads verify, both maintainers can operate it, and the serving path does not depend on Triton. Unknowns and exclusions must be reported rather than described as completed preservation.

## Hosting and cost

R2 Standard is the leading candidate, subject to the full census. Its current monthly included usage is 10 GB-month storage, 1 million Class A operations and 10 million Class B operations; egress is free. This is an allowance with billable overages, not a guaranteed spending cap or a promise that pricing will never change. [Cloudflare R2 pricing](https://developers.cloudflare.com/r2/pricing/) (checked 15 September 2026).

Target zero recurring hosting cost within measured allowances. Record headroom, usage alerts, ownership of billing and the response to approaching limits. Do not auto-upgrade, silently prune preserved data, or treat alerts as hard enforcement. Keep a provider-independent archive format, independent backup and migration instructions so a pricing change does not end preservation. Object copies for rollback, model weights and bulk-download bundles must be included in the budget if stored in the same account.

## Security and shared maintenance

- Chris and Oliver have separate Cloudflare identities with MFA and appropriate access to the same owning account; no shared personal password or dependency on one person's laptop. Document account recovery and ownership transfer in a shared secure vault.
- Separate public synthetic storage from private staging/backups. Public accessibility is intentional; publication classification and write access are the security boundaries.
- Use bucket-scoped object permissions for automation wherever supported. Keep provisioning/admin permissions separate from routine upload/read jobs. Record effective permissions and token owner/rotation procedure; do not claim write-only enforcement if the provider's grant also permits deletes. [R2 authentication and token scopes](https://developers.cloudflare.com/r2/api/tokens/).
- Put credentials in protected CI secrets or the maintainers' vault, never git, browser bundles, URLs or log output. Fork/untrusted PR jobs must not receive publication credentials. Record credential references in the repo, not secret values.
- Updates create immutable release prefixes and verify checksums before moving a small release pointer. Rollback selects a verified previous manifest. Avoid destructive sync/delete flags in normal publishing; unexpected removals require review.
- Archive format, manifests and scripts live in the repo. Operational access instructions must work for either maintainer. Keep private source mappings outside public documentation.

## As-built record to complete during execution

### Initial inventory tool — 15 September

`tools/cache_inventory.py` now performs a bounded, read-only metadata census with strict current-source seed/hash shard rules and optional public-name coverage. Run from the repo with `python3 tools/cache_inventory.py --root active=/absolute/cache/root --seconds 30 --names-file /path/to/catalogue.json`. Four unit tests pass (`python3 -m unittest discover -s tools -p test_cache_inventory.py`). It never reads pixels or approves publication. Other path conventions remain unresolved; a deadline-limited result cannot establish missing names or total archive size.

The first read-only Triton sample exhausted 30 seconds in the active tree: 2,367 seed-path candidates, 73,747,385 bytes; oldSeeds was not reached. A resumable/chunked full census is still required. Evidence is kept in the workspace `review-artifacts/cache-census-2026-09-15/`. This is tool validation and partial metadata evidence, not completed preservation.

| Item | Current evidence |
|---|---|
| Active + oldSeeds inventory and classification revision | Pending; earlier oldSeeds census is partial migration evidence only |
| Export tool revision, exact commands and prerequisites | Pending implementation |
| Included/excluded/unresolved counts and bytes | Pending complete census |
| Public manifest schema, release ID and checksum | Pending |
| Provider/account, bucket, hostname and cache/CORS configuration | Not provisioned by this work |
| CI workflow, secret names, maintainer roles and rotation | Pending |
| Upload verification, names coverage and bulk-download test | Pending |
| Independent backup location and restore rehearsal | Pending; location recorded in shared private operations records |
| Chris and Oliver maintenance/recovery rehearsals | Pending |
| Measured free-tier headroom and usage-alert owner | Pending |
| Final delta, cutover and rollback evidence | Pending separate live-action scope |

Record dated evidence and exact reproducible commands here as work completes. The migration's independent history-preservation, public comparison and scoped cutover gates still apply.

## 16 September — selected Cloudflare collection versus bulk preservation

The operator reports roughly **2 TB in the Triton directory**. This is an operator estimate of the mixed source tree, not a measured public-image dataset. Do not bulk-upload that directory or assume it fits free hosting.

The initial Cloudflare Static Assets publication is **only a selected allowlist of proven seed-generated still images**, including names only where the deterministic synthetic mapping is verified. Never publish recovered/uploaded/e4e photo outputs, registered arbitrary latents, mixed-photo results, or unknown provenance. A filename, numeric-looking identifier, hash, or plausible-looking face is not sufficient provenance. No public user-upload endpoint. Upload tools must resolve the reviewed manifest and verify hashes; do not recursively sync a Triton cache root.

Inventory before deciding capacity: measure counts/bytes by source root, provenance class, still image versus video/intermediate, resolution, format, model era and unique checksum. Use bounded/resumable read-only metadata passes first; hash only classified candidates in a controlled export stage. Do not read/copy/hash2 TB in one unbounded production scan. Report unreadable/changed/unknown files separately; a sampled census is not a total.

Prioritize the complete verified names collection and an explicitly curated common-seed set, then expand using legitimate aggregate popularity evidence if available. Keep one canonical **actual1024 lossless original** per new generation identity/model/noise configuration, with only useful display derivatives. Old small JPEGs stay historic artifacts; neither upscaling nor PNG conversion makes them original1024 renders. Generate a missing1024 version offline/on a qualified nonproduction route if needed, with separate provenance and licensing review.

Public fast cache, complete eligible-history preservation and private retention are different sizing decisions. The existing full eligible-history preservation objective remains pending census; a small Cloudflare release neither fulfills it nor authorizes deletion of the remainder. Once counts/bytes are known, present the measured options for the long tail, backups and private records before choosing hosting or changing retention. No paid storage or automatic overage is authorized. Cache misses use qualified device generation; repeat requests reuse the local1024 original.

## 16 September — measured selection and names request identity

See the [read-only Triton investigation](../../review-artifacts/static-assets-2026-09-16/README.md). The names bundle lowercases catalogue values before requesting images; preserve this exact request mapping as well as catalogue spelling/order. The existing inventory helper hashes supplied names verbatim, so its original uppercase input is not a valid names coverage check. Use the derived `names-request-identities.json` and completed corrected probe evidence for this candidate. Numeric seed paths and name-derived paths are deterministic generation candidates; broad hash directories are not an upload allowlist.
