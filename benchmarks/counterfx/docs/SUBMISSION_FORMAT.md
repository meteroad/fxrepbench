# Submission format

`fxrepbench evaluate` accepts JSON Lines or CSV. Each row describes one rendered
candidate:

| Field | Required | Meaning |
| --- | --- | --- |
| `item_id` | yes | Frozen benchmark item ID. |
| `audio_path` | yes | Absolute path or path relative to the submission file. |
| `candidate_id` | no | Unique candidate label within the item. |
| `score` | for pools | Representation score; larger values rank higher. |
| `chain_order` | no | JSON list or comma-separated effect IDs for FX-set F1. |

For ordinary system evaluation, provide exactly one row per item. No `score` is
needed. For fixed-pool representation evaluation, provide multiple rows per
item and a score for every candidate. The evaluator reports:

- selected `L_d`: target error of the highest-scoring candidate;
- oracle `L_d`: lowest target error available in the same pool;
- selected--oracle gap: ranking error within that pool;
- FX-set F1 for the selected candidate when `chain_order` is supplied.

Example JSONL:

```json
{"item_id":"fma_val_125159","candidate_id":"000","audio_path":"audio/fma_val_125159/000.wav","score":0.82,"chain_order":["compressor","reverb"]}
{"item_id":"fma_val_125159","candidate_id":"001","audio_path":"audio/fma_val_125159/001.wav","score":0.77,"chain_order":["peak_filter","reverb"]}
```

Audio must be stereo, 44.1 kHz, finite, and exactly 10 seconds. The evaluator
does not resample, remix, pad, or trim submissions.

```bash
fxrepbench evaluate \
  --track counterfx-dev80 \
  --submission submission.jsonl \
  --data-root /path/to/fma \
  --output-dir evaluation/dev80
```

The output directory contains per-candidate results, per-item selected/oracle
results, and a machine-readable summary.
