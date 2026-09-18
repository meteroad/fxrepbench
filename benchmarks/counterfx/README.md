# CounterFX

CounterFX is the first benchmark family in FXRepBench. It evaluates controlled
added-processing transfer and representation-based ranking against an exactly
reachable, same-content target.

## Tracks

| Track | FMA split | Items | Purpose |
| --- | --- | ---: | --- |
| `counterfx-dev80` | validation | 80 | Development and integration checks |
| `counterfx-200` | test | 200 | Frozen paper evaluation |

For each item, the same sampled effect chain is applied to adjacent source and
reference windows:

```text
A_source = A
B_ref    = R(B, target_chain)
A_target = R(A, target_chain)
```

The system receives `A_source` and `B_ref`; `A_target` is reserved for
evaluation. CounterFX supports output-level transfer evaluation and fixed-pool
analysis that separates candidate coverage from representation ranking.

## Contents

```text
configs/            Processor library, priors, metric, and renderer settings
manifests/          Frozen Dev80 and CounterFX-200 item definitions
docs/               Benchmark card, protocol, and submission format
locks/              Pedalboard and CMA-ES dependency locks
reference_results/  Aggregate paper curves without audio or checkpoints
```

- [Benchmark card](docs/BENCHMARK_CARD.md)
- [Frozen protocol](docs/PROTOCOL.md)
- [Submission format](docs/SUBMISSION_FORMAT.md)
- [Manifest description](manifests/README.md)
- [Reference results](reference_results/README.md)

Raw FMA audio is not redistributed. Use the repository-level `fxrepbench` CLI
to validate manifests, reconstruct items, generate shared candidate pools, and
evaluate submissions.
