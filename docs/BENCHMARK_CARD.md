# FXRepBench benchmark card

## Intended use

FXRepBench evaluates audio-effect representations and executable effect-transfer
systems. Its first track, CounterFX-200, provides exactly reachable targets for
controlled added-processing transfer and supports fixed-pool analysis of
representation-based ranking.

## CounterFX-200 task

For each item, two adjacent, non-overlapping 10-second windows are extracted
from the same 30-second FMA-small excerpt. Both windows are normalized to
-20 LUFS. A sampled one-to-four-effect chain is applied to the second window to
form the processed reference and to the first window to form the withheld
same-content target.

The system observes only the unprocessed first window and processed second
window. The unprocessed counterpart of the reference is withheld.

## Evaluation views

### Transfer output

The primary system output is rendered audio for the source content. Output
error is measured against the exactly constructed target. This evaluates the
resulting sound rather than recovery of one unique generating chain.

### Fixed-pool representation ranking

Representations rank a shared set of rendered candidates. Oracle error measures
candidate coverage. The difference between selected and oracle error measures
ranking error. Candidate generation and representation ranking must be reported
separately.

## Data

- Development: 80 FMA-small validation excerpts, 10 per genre.
- Test: 200 FMA-small test excerpts across eight genres.
- Audio: stereo, 44.1 kHz, two 10-second windows per item.
- Processing: one to four effects selected from eight effect types, with 22
  controls in the complete processor library.

The release contains metadata and deterministic reconstruction instructions,
not the source recordings.

## Current scope

CounterFX-200 measures controlled, in-library added-processing transfer. It
does not by itself establish perceptual matching for arbitrary natural
cross-recording references or generalization to unseen renderer families.
The accompanying listening study assesses the perceptual relevance of the
paper metric within the controlled setting; its ratings are not currently a
training set or leaderboard target.

## Known limitations

- The current processor coverage contains eight Pedalboard effects.
- Target salience is filtered with the same output distance later reported by
  the benchmark.
- Public target manifests permit exact reconstruction and are therefore not a
  sequestered challenge test set.
- FMA uses lossy 30-second source excerpts and per-track licenses.

Future versions may add separate cross-renderer and perceptual-transfer tracks
without changing CounterFX-200.

