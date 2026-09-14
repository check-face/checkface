# Original Facemorph UI source overlay

This builds the existing F# / Fable / Elmish / React application for the local
CheckFace API. It is not a replacement page or new UI. The source is pinned in
`source.json` to `check-face/facemorph.me` commit
`0abb215f27b16e17e3919cf78b616b6ae4998a5d`.

Apply the files under `source/` over that exact checkout. Keep `package-lock.json`
and `.config/dotnet-tools.json` from the pinned source; the overlay changes no npm
or Fable dependency versions. Do not copy the migration workspace wholesale:
research outputs, models, private inputs and unfinished Next/desktop work are
not needed by this build.

## Container source build

Use the official .NET 5.0.408 bullseye SDK **only in the frontend build stage**,
with Node 22. The verified SDK manifest includes Linux amd64, arm64 and arm/v7:

`mcr.microsoft.com/dotnet/sdk:5.0.408-bullseye-slim@sha256:40c6bd0059eaa06b4a9c91cd3e6df138f6224bd02b2882bf6ce3aa4af3835fc5`

The overlay selects SDK `5.0.408` with latestPatch and keeps Fable 3.1.12.
.NET 5 is an EOL build tool; it is pinned here to compile the existing application,
not installed in the final runtime. A tested attempt with SDK 8 and major runtime
roll-forward restored packages but the old Fable project loader failed to resolve
its references. Upgrading that compiler/toolchain is separate from reproducing the
existing UI. The final server remains the modern Python/PyTorch image and receives
only static frontend output. The main frontend repository's historical SDK
selection is unchanged.

From the overlaid frontend source directory:

```sh
npm ci
npm run build:self-host
```

The script restores the pinned dotnet tool, compiles `src` and invokes the
existing webpack configuration for the client only. Copy `deploy-self-host/`
into the API image's `/app/frontend/`. No Vercel server bundle or npm runtime is
needed by the API container. Python serves `index.html`, static assets and
extensionless SPA paths such as `/retirement`; its `/api/...` endpoints take
precedence over the SPA fallback. Root-mounted hosting is required because the
original page uses `<base href="/">` and root-relative assets.

The source build is deterministic in dependency selection, not yet a claim of
bit-identical bundles. Docker build and browser smoke evidence belong to the
self-host candidate validation, not this overlay alone.

## Deliberate configuration differences

`FACEMORPH_SELF_HOST=1` is set by the build script. The public pinned source
contains the classic UI only, with no HF/trial code; inherited trial/review
variables are also cleared by the script. The overlay Config is derived from
that public classic source rather than the workspace's unpublished trial branch. All original face,
slider, video, upload and preview requests use the same origin's legacy API
paths. Canonical/share links use the current page origin, never the production
service. The static template omits production Google Analytics.

The original Browse dialog uses its existing Values/RenderValue/OnValueSelected
contract to show the supplied seed choices locally. It does not embed the
separate names.facemorph.me service. The hosted build retains its original iframe
behavior. Self-host images use native lazy loading so opening the gallery does
not immediately queue every offscreen seed for CPU inference. Existing text, seed, upload, morph/slider and download controls are
otherwise unchanged. The frontend does not make HF/trial inference requests or
load a new client inference engine in this mode.

Production builds without `FACEMORPH_SELF_HOST=1` keep their prior defaults.
External documentation and user-initiated social-share links remain ordinary
links; they are not inference endpoints. A local URL is only useful to someone
who can reach that local server.

## Provenance

`source.json` records the upstream commit and SHA-256 of each overlay file.
Retain the pinned source's existing notices, assets and dependency licenses.
That frontend commit contains no top-level LICENSE file; this record does not
invent or extend redistribution rights. No prebuilt image or compiled frontend
release is published by this overlay.

The original analytics helper now guards absent `gtag`; removing analytics must
not interrupt Morph, uploads, sliders or sharing. The self-host homepage omits the
public retirement banner (including its historical HF proposal), which does not
apply to this local API. Hosted builds retain the original banner.
