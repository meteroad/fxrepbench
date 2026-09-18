# Frozen manifests

Each track contains:

- `audio.csv`: FMA identity, license, source hash, segment offsets, and signal
  statistics;
- `targets.jsonl`: target chain, physical and normalized controls, renderer
  metadata, and target-audibility statistics;
- `summary.json`: aggregate construction statistics;
- `rejected_audio.jsonl`: source items rejected before target generation; and
- `rejected_targets.jsonl`: rejected parameter draws, including the reason.

The original manifest identifiers are preserved to maintain exact provenance
with the paper experiments. `pedalboard-fma-dev-v0.3` maps to the public
`counterfx-dev80` track, and `pedalboard-fma-test-v0.1` maps to
`counterfx-200`.

The target metadata is public for reproducibility. It is evaluation metadata
and must not be used as input to a benchmarked system.
