# CounterFX protocol v1

## Frozen inputs

CounterFX uses the official FMA-small validation and test splits. Eligible
items must have a declared Creative Commons or public-domain license that does
not contain a NoDerivatives restriction. Items are ranked deterministically by
a SHA256 value derived from the frozen audio seed.

The public suite contains:

| Track | FMA split | Items | Purpose |
|---|---:|---:|---|
| `counterfx-dev80` | validation | 80 | development and integration checks |
| `counterfx-200` | test | 200 | paper evaluation |

The frozen audio seed is `20270720`; the effect seed is `20270722`.

## Audio construction

1. Decode the complete FMA-small excerpt.
2. Preserve stereo, duplicating mono sources only when necessary.
3. Resample to 44.1 kHz if required.
4. Extract adjacent, non-overlapping 10-second windows beginning at the frozen
   manifest offset.
5. Normalize each window independently to -20 LUFS.

The preferred windows are `[5, 15)` and `[15, 25)` seconds. If they fail the
predefined audio validity checks, the construction searches deterministic
adjacent 20-second windows without examining model performance.

## Target processing

Chains contain one to four of the following effects:

1. high-pass filter;
2. low-pass filter;
3. peak filter;
4. compressor;
5. distortion;
6. chorus;
7. delay; and
8. reverb.

The effect and parameter manifests freeze effect identities, chain order,
normalized controls, physical controls, seeds, renderer version, and all
rejected target draws. Physical controls in `targets.jsonl` are authoritative
for reconstruction. The legacy normalized fields are retained for provenance;
new systems must derive prior-aware normalized controls from `configs/effects.json`.

The reference renderer is Spotify Pedalboard 0.9.23 at 44.1 kHz with an
8192-sample buffer, processor reset enabled, and no peak limiting or RMS
matching.

## Topology settings and candidate pools

In **known topology**, effect identities and order are supplied while continuous
controls remain hidden. In **hidden topology**, chain length, identities, order,
and controls are all hidden.

The shared Global Random pool uses candidate seed `20270724`. For each item and
topology, an item seed is derived with SHA256. Hidden-topology chains follow the
independent activation probabilities and four equiprobable order templates in
`configs/chain_prior.json`; known-topology candidates use the supplied chain.
Non-finite or peak-clipping candidates are rejected and replaced until the
requested number of valid candidates is reached. Candidate budgets are prefixes
of the same deterministic pool.

## Salience filter

Before any method is evaluated, target parameters are resampled until both
source-to-target and reference-pair distances satisfy `Ld >= 0.8`. Non-finite
or clipped renders are rejected. Rejections are retained in the release.

## Metrics

`L_d` is the arithmetic mean of spectral-convergence and log-magnitude losses
at three STFT resolutions. Its frozen FFT, hop, window, epsilon, and aggregation
settings are recorded in `configs/metric.json` and implemented in
`fxrepbench.metrics.MultiResolutionStftDistance`.

For a ranked candidate pool, selected `L_d` evaluates the highest-scoring
candidate, oracle `L_d` is the lowest error present in that same pool, and their
difference is the ranking gap. FX-set F1 compares unordered effect sets and
ignores order and continuous controls.

## Information boundaries

A benchmark system receives `A_source` and `B_ref`. It must not use the target
manifest, target chain, target parameters, `B`, or `A_target` as model input.
The public target manifest exists for transparent reconstruction and audit.

For future leaderboard use, a separately generated hidden split should retain
the same construction while withholding target metadata.
