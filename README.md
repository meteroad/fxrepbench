# FXRepBench

**An extensible benchmark suite for audio-effect representations.**

FXRepBench evaluates whether an audio-effect representation can support
executable effect transfer and rank rendered candidates by their processing
match. The suite separates controlled, exact-target evaluation from future
cross-renderer and perceptual tracks.

This repository is currently a **v0.1.0 pre-release**. The first included track
is CounterFX-200, accompanied by an 80-item development split.

## Available tracks

### CounterFX-200

CounterFX-200 is a controlled benchmark for added-processing transfer. Each
item takes two adjacent, non-overlapping windows, `A` and `B`, from one FMA-small
excerpt and applies the same hidden effect chain to both:

```text
A_source = A
B_ref    = R(B, target_chain)
A_target = R(A, target_chain)
```

A system receives `A_source` and `B_ref`. The exactly reachable,
same-content `A_target` is used only for evaluation. This construction retains
the recording's existing production state and does not require an
effect-neutral input.

The current release supports two evaluation views:

- **Transfer:** compare a system output with `A_target`.
- **Fixed-pool ranking:** give representations the same rendered candidates and
  separate candidate coverage from ranking error.

See [the CounterFX benchmark directory](benchmarks/counterfx/README.md) for its
scope, frozen protocol, manifests, and reference results.

## Repository layout

```text
benchmarks/
└── counterfx/              CounterFX configs, manifests, docs, and results
src/fxrepbench/             Shared reconstruction and evaluation utilities
docs/                       Suite-level release documentation
tests/                      Release-integrity tests
```

Raw FMA audio is not included. The manifests provide FMA track IDs, source
URLs, per-track licenses, segment offsets, and source-file SHA256 hashes. Users
obtain FMA-small separately and reconstruct benchmark audio locally.

## Installation

Manifest inspection and validation use only Python's standard library:

```bash
python -m pip install -e .
fxrepbench info
fxrepbench validate
```

Install the locked reference dependencies to reconstruct items:

```bash
python -m pip install -e '.[reference]'
```

The strict reference path uses PyTorch/Torchaudio because changing the MP3
decoder causes small but measurable differences in reconstructed targets.

Arrange the original FMA-small archive under a data root:

```text
/path/to/data/
└── fma/
    └── fma_small/
        ├── 000/
        ├── 001/
        └── ...
```

Then reconstruct one development item:

```bash
fxrepbench render-item \
  --track counterfx-dev80 \
  --item-id fma_val_125159 \
  --data-root /path/to/data \
  --output-dir rendered/fma_val_000002
```

The command verifies the source MP3 hash before decoding and writes
float32 `source.wav`, `reference.wav`, and `target.wav` files. Float output
avoids introducing quantization error into metric validation.

Reconstruct a complete local track with:

```bash
fxrepbench prepare \
  --track counterfx-200 \
  --data-root /path/to/data \
  --output-dir rendered/counterfx-200
```

## Evaluation

The evaluator supports both one output per item and ranked candidate pools:

```bash
fxrepbench evaluate \
  --track counterfx-dev80 \
  --submission submission.jsonl \
  --data-root /path/to/data \
  --output-dir evaluation/dev80
```

It reports selected and oracle `L_d`, their gap, and optional FX-set F1 using
the exact metric definition used in the paper. See
[`benchmarks/counterfx/docs/SUBMISSION_FORMAT.md`](benchmarks/counterfx/docs/SUBMISSION_FORMAT.md)
for the schema.

Generate the deterministic shared Global Random pool used to compare
representation-based rankers:

```bash
fxrepbench generate-random-pool \
  --track counterfx-dev80 \
  --topology hidden \
  --budget 512 \
  --data-root /path/to/data \
  --output pools/dev80-hidden-512.jsonl
```

Add `--audio-dir pools/audio` to retain float32 candidate renders. Without it,
the command still renders every draw to apply the frozen finite-value and
clipping checks, but stores only the compact executable chain definitions.

The public baseline modules also include the exact two-hidden-layer
registration MLP, Head-centered proposal decoding, and a renderer-agnostic
CMA-ES ask/tell loop. Install `.[baselines]` to run the latter. Encoder adapters
and approved checkpoints remain separate because their redistribution terms
must be confirmed per model.

## Reproducibility

The frozen manifests record:

- item and effect seeds;
- FMA split, track identity, license, timestamp, and source hash;
- effect identity, order, and physical parameter values;
- rejected source items and target draws;
- loudness normalization and renderer configuration.

Run the release checks with:

```bash
python -m unittest discover -s tests -v
```

Reconstruct the paper's main candidate-pool table from the frozen CSVs with:

```bash
python tools/summarize_reference_results.py
```

## Planned extensions

Future releases may add cross-renderer robustness and perceptual-transfer
tracks. These will be versioned independently; CounterFX-200 will remain a
frozen track.

## Citation and licenses

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). This
repository uses Apache-2.0 for source code and CC BY 4.0 for original benchmark
metadata and documentation. Original FMA recordings remain governed by their
per-track licenses; see [`LICENSES.md`](LICENSES.md).
