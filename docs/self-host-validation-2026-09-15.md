# Self-host follow-up validation — 15 September 2026

[Linux AMD64 clean-source CI passed](https://github.com/check-face/checkface/actions/runs/34940400054) for commit `264356b3be29da07491ace0ee5839e382358f68d` on `candidate/self-host-validation-20260915`.

The GitHub-hosted Ubuntu 24.04 runner built its own CPU image, acquired checksum-verified model assets on the runner, and exercised actual inference. Auto configuration selected four model threads from four available CPUs. Eleven CPU configuration tests, ten real PNG cache/storage tests and two real-tensor adapter tests passed. Eight preserved-route contract tests and three encoder-asset tests also passed, alongside the original UI, actual e4e encoding/alignment, retained API/media/upload and restart workflows.

Three concurrent requests for the same new seed at different dimensions/formats performed **one synthesis and two original-cache hits**, with zero write failures. After restarting the API, two previously unrequested dimensions performed **zero synthesis and two retained-original hits**, again with zero write failures. Lightweight test excerpts and the machine-readable evidence record are in the migration workspace at `review-artifacts/delivery-followup-2026-09-15/selfhost/`.

This validates the integrated private full-resolution cache and CPU self-host candidate on this runner. It does not establish GPU or native desktop support, historic byte parity, a performance optimum across CPU allocations, large-cache scale or model-free cache startup. The server still eagerly initializes its model. Original PNGs can include photo-derived faces and are not the synthetic-only public preservation archive.

Only source was pushed. No model weights, container image, release artifact or live deployment was published. Later documentation and read-only inventory-tool commits do not change the tested self-host source revision above.
