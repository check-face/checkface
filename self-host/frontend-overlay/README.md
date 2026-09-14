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

Use the .NET 8 bookworm SDK image and Node 22 bookworm toolchain. The source-only
overlay selects SDK 8 (`8.0.100`, latestFeature); set `DOTNET_ROLL_FORWARD=Major`
so the pinned Fable 3.1.12 tool, originally targeting .NET 5, can run on the
installed runtime. The main frontend repository's historical SDK selection is
unchanged. From the overlaid frontend source directory:

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
