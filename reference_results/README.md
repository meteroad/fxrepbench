# Frozen paper results

These CSV files are the item-aggregated candidate-budget curves used for the
CounterFX-200 experiments. They are included as immutable reference outputs so
that tables and plots can be checked without redistributing FMA audio or model
checkpoints.

- `counterfx-200/candidate_budget/` contains Global Random, Head@1, and
  Head-centered curves for RelFx, Fx-Encoder++, and AFx-Rep.
- `counterfx-200/cmaes/` contains the four known-topology and three
  hidden-topology RelativeFx CMA-ES runs.

The CSVs contain aggregate metrics only. They do not contain private audio,
internal training metadata, absolute paths, or participant information.
`tools/summarize_reference_results.py` reconstructs the main comparison table
from these curves.
