# First Result Memo

This file is a scaffold for Deliverable F.

Regenerate it after collecting benchmark runs with one of:

```bash
uv run python -m wdlm.cli compare-benchmark-runs --runs runs/toy_baseline,runs/toy_state_head_baseline,runs/toy_para_align_baseline,runs/toy_wdlm --out-dir runs/core_comparison --memo-out docs/first_result_memo.md
uv run python -m wdlm.cli run-ablation-suite --include-core-models
```

Expected sections after regeneration:

- experiment list
- best metrics by model
- biggest OOD drops
- provisional interpretation placeholders
- questions to answer manually
