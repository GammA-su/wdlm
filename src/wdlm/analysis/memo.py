"""Generate a first-result memo scaffold from benchmark comparison outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any


OOD_SPLITS: tuple[str, ...] = (
    "test_lexical_ood",
    "test_paraphrase_ood",
    "test_compositional_ood",
    "test_length_ood",
)


def _best_run_for_metric(payload: dict[str, Any], split_name: str, metric_name: str) -> tuple[str, float]:
    best_entry = payload["best_by_split_metric"][split_name][metric_name]
    run_name = str(best_entry["run_name"])
    value = 0.0 if best_entry["value"] is None else float(best_entry["value"])
    return run_name, value


def _biggest_ood_drop(run_row: dict[str, Any]) -> tuple[str, float]:
    iid = float(run_row["splits"]["test_iid"]["exact_state_accuracy"])
    drops = {
        split_name: iid - float(run_row["splits"][split_name]["exact_state_accuracy"])
        for split_name in OOD_SPLITS
    }
    split_name = max(sorted(drops), key=lambda key: drops[key])
    return split_name, float(drops[split_name])


def render_first_result_memo(payload: dict[str, Any]) -> str:
    """Render a first-result memo scaffold from aggregate comparison data."""

    lines = [
        "# First Result Memo",
        "",
        "## Experiment List",
        "",
    ]
    for run in sorted(payload["runs"], key=lambda row: row["run_name"]):
        lines.append(f"- `{run['run_name']}`")
    lines.extend(
        [
            "",
            "## Best Metrics By Model Family",
            "",
        ]
    )
    for split_name in ("val", "test_iid"):
        best_exact_run, best_exact_value = _best_run_for_metric(payload, split_name, "exact_state_accuracy")
        best_para_run, best_para_value = _best_run_for_metric(payload, split_name, "paraphrase_delta_cosine")
        lines.append(f"- `{split_name}` best exact-state accuracy: `{best_exact_run}` at `{best_exact_value:.4f}`")
        lines.append(f"- `{split_name}` best paraphrase delta cosine: `{best_para_run}` at `{best_para_value:.4f}`")
    lines.extend(
        [
            "",
            "## Biggest OOD Drops",
            "",
        ]
    )
    for run in sorted(payload["runs"], key=lambda row: row["run_name"]):
        split_name, drop_value = _biggest_ood_drop(run)
        lines.append(f"- `{run['run_name']}` largest exact-state drop: `{split_name}` by `{drop_value:.4f}`")
    lines.extend(
        [
            "",
            "## Provisional Interpretation",
            "",
            "- Placeholder: Does WDLM beat the matched baselines on `test_iid` exact-state accuracy?",
            "- Placeholder: Which model retains the strongest exact-state accuracy under lexical and paraphrase OOD?",
            "- Placeholder: Does removing `L_para`, `L_state`, `L_ctr`, or state conditioning collapse the WDLM advantage?",
            "- Placeholder: Are text-loss differences small enough that state-tracking gains are the main story?",
            "",
            "## Questions To Answer Manually",
            "",
            "- Which comparison should anchor the claim: best WDLM vs best non-WDLM baseline, or WDLM vs each baseline separately?",
            "- Are the OOD gaps qualitatively aligned across lexical, paraphrase, compositional, and length settings?",
            "- Which ablation is the decisive kill test for the WDLM hypothesis on toy world?",
            "- Do any wins appear driven mostly by explicit state supervision rather than delta transition modeling?",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def write_first_result_memo(payload: dict[str, Any], *, out_path: Path) -> Path:
    """Write the first-result memo scaffold to disk."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_first_result_memo(payload), encoding="utf-8", newline="\n")
    return out_path
