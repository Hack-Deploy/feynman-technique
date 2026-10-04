# Hugging Face replay examples

Source: [discovery-market-live](https://huggingface.co/datasets/arushisinha98/discovery-market-live), revision `d240272bae837eb24afa2089f31bc6ef4de4388e` (4 October 2026).

The 21 runs contain 10 successes and 11 failures. Success means the original recorded verdict passed; the viewer never re-scores old runs using the current claim configuration.

For each outcome, select a run with measured curves, then favor more measured points, more individual curves, and recorded fits. Missing outcomes remain unavailable. A failure with a useful experiment can be a better teaching example than a run that withdrew before measuring anything.

| Claim | Success example | Failure example |
|---|---|---|
| gravity-inverse-square | Claude Sonnet 5.5 — 3 paths with multiple readings, 15 points (`e0d7106a-e12e-5267-b0cd-5ccf1e62e5d1`) | Claude Sonnet 5 — 3 paths with multiple readings, 15 points (`ff82088a-ed34-54ef-b744-d725077149f2`) |
| oscillator-time-varying | Claude Sonnet 5.5 — 2 paths with multiple readings, 20 points (`32988fa0-ad00-5b92-9108-e08584364e0f`) | Claude Sonnet 5 — 3 paths with multiple readings, 20 points (`bbae52eb-6033-5987-ad8e-50be5adca601`) |
| dark-matter-unseen-pull | No recorded example | Claude Sonnet 5 — 5 paths with multiple readings, 50 points (`130172f3-a10b-5896-b253-027dba579639`) |
| yukawa-screened | Claude Sonnet 5.5 — 4 paths with multiple readings, 12 points (`95c6b13c-deb6-5ff8-a79a-9bf3b81617e7`) | Claude Haiku 4.5 — 4 paths with multiple readings, 12 points (`8c80c2e9-cc1a-5761-8d06-28ea938c16f7`) |
| coulomb-source-strength | Claude Sonnet 5 — 0 paths with multiple readings, 2 points (`ac6488be-2af1-5c28-ac7a-fdb5b12c1cc6`) | No recorded example |
| fractional-p2-inertia | Claude Haiku 4.5 — 3 paths with multiple readings, 18 points (`1cca550b-487e-51c1-84c8-f97b34f1c8a2`) | No recorded example |
| extra-dimensions-short-range | Claude Sonnet 5.5 — 12 paths with multiple readings, 30 points (`6641cf26-25a9-5e13-b1c7-721ad1b87c99`) | Claude Opus 5.5 — 0 paths with multiple readings, 0 points (`4bae2b81-a86d-583c-ab31-0513b6d65428`) |
| circle-ordinary-gravity | Claude Haiku 4.5 — 20 paths with multiple readings, 200 points (`3d8102a1-6b45-55d3-bce2-6dac076cfeb1`) | No recorded example |
| three-species-repulsion | Claude Sonnet 5.5 — 10 paths with multiple readings, 40 points (`c72be55b-b50e-5627-a49c-44e8a7da7680`) | No recorded example |
| ether-outward-push | Claude Sonnet 5 — 10 paths with multiple readings, 20 points (`091175de-5995-5704-960c-56de7ab8fb22`) | No recorded example |
| hubble-outward-push | Claude Opus 5.5 — 10 paths with multiple readings, 40 points (`41768fcd-6e1f-5200-8274-2e301d8a77ff`) | No recorded example |

The extra-dimensions failure withdrew without experiments: it is the only failure available for that claim. Coulomb has two single-time observations, so it is shown as dots rather than invented time curves.

All selection routes use the same viewer: success/failure tabs, claim/model picker, recorded-run table, and live jobs. Motion and particle-path plots retain each probe separately. Lines connect observed samples; they do not add measurements. The matched gravity pair additionally supports the existing force-law comparison, with derived pull readings and explicitly labeled simulator reference curves. Other claims are not forced into a power-law interpretation.

No numeric estimate is fabricated when the model did not record one. Recorded fit parameters remain visible even when the final verdict is wrong.
