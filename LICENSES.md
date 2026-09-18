# Licensing

FXRepBench uses separate licenses for software and benchmark materials.

## Source code

Source code is licensed under the Apache License 2.0. The full terms are in
[`LICENSE`](LICENSE). This applies to:

- `src/`;
- `tests/`;
- `tools/`;
- `pyproject.toml`; and
- other executable source files unless a file states otherwise.

SPDX identifier: `Apache-2.0`.

## Benchmark metadata and documentation

Original benchmark manifests, configurations, aggregate reference results, and
documentation are licensed under the Creative Commons Attribution 4.0
International License. The full terms are in
[`LICENSE-DATA`](LICENSE-DATA). This applies to:

- `benchmarks/`;
- `docs/`;
- repository documentation and citation metadata.

SPDX identifier: `CC-BY-4.0`.

Please attribute these materials to the FXRepBench authors and cite the release
using [`CITATION.cff`](CITATION.cff).

## FMA source audio

FXRepBench does not redistribute FMA audio. Each recording remains subject to
the license selected by its creator. The Creative Commons license above covers
only FXRepBench's original compilation and annotations; it does not replace,
extend, or override any source recording's license.

The audio manifests preserve the license title, license URL, source URL, and
source-file SHA256 for every item. Users are responsible for obtaining
FMA-small from an authorized source and complying with the applicable per-track
terms.

## Third-party software and models

Spotify Pedalboard, ST-ITO/CMA-ES, FMA, and referenced representation models
remain under their respective licenses. They are dependencies or provenance
references and are not relicensed by this repository. Model checkpoints are
not included unless a future release explicitly identifies their redistribution
terms.
