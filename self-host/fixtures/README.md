# Fixed CPU references

Synthetic seed 0 and 1 images copied unchanged from migration workspace
`review-artifacts/stylegan-proof/seed-{0,1}-fresh.png`, recorded 9 September 2026.
Original evidence: CPU PyTorch 2.14.0, converted deployed config-F generator,
constant noise, truncation psi 0.7/cutoff 8, full synthesis then 512px Lanczos.
The image files are test fixtures, not learned model weights or uploaded photos.

Smoke renders 1024px and compares every RGB channel after the same resize to
these independent references, allowing maximum error 1. It also checks full
1024px repeatability and morph endpoint identity. This is converted-runtime
regression evidence, not equality to historic JPEG files or all 31 research cases.
