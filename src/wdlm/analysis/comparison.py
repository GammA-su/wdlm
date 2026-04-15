"""Aggregate and compare benchmark evaluation outputs across runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wdlm.utils.io import stable_json_dumps


COMPARISON_SPLITS: tuple[str, ...] = (
    "val",
    "test_iid",
    "test_lexical_ood",
    "test_paraphrase_ood",
    "test_compositional_ood",
    "test_length_ood",
)

COMPARISON_METRICS: tuple[str, ...] = (
    "text_loss",
    "exact_state_accuracy",
    "per_field_accuracy",
    "paraphrase_delta_cosine",
)


@dataclass(frozen=True)
class RunBenchmarkData:
    """Resolved benchmark evaluation files for one run."""

    run_name: str
    run_dir: Path
    benchmark_dir: Path
    per_split_metrics: dict[str, dict[str, Any]]
    aggregate_metrics: dict[str, Any]


def _resolve_benchmark_dir(run_dir: Path) -> Path:
    candidate = run_dir / "benchmark_eval"
    if candidate.exists():
        return candidate
    return run_dir


def _load_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Expected JSON object in {path}.")
    import json

    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return payload


def load_run_benchmark(run_dir: Path) -> RunBenchmarkData:
    """Load benchmark evaluation outputs from a run directory."""

    benchmark_dir = _resolve_benchmark_dir(run_dir)
    per_split_path = benchmark_dir / "per_split_metrics.json"
    metrics_path = benchmark_dir / "metrics.json"
    if not per_split_path.exists():
        raise FileNotFoundError(f"Missing benchmark eval file: {per_split_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing benchmark eval file: {metrics_path}")
    per_split_metrics = _load_json(per_split_path)
    aggregate_payload = _load_json(metrics_path)
    aggregate_metrics = aggregate_payload.get("aggregate_metrics", aggregate_payload)
    run_name = run_dir.name
    return RunBenchmarkData(
        run_name=run_name,
        run_dir=run_dir,
        benchmark_dir=benchmark_dir,
        per_split_metrics=per_split_metrics,
        aggregate_metrics=aggregate_metrics,
    )


def build_comparison_payload(run_dirs: list[Path]) -> dict[str, Any]:
    """Build a deterministic comparison payload from multiple runs."""

    runs = [load_run_benchmark(path) for path in sorted(run_dirs, key=lambda item: item.name)]
    comparison_rows: list[dict[str, Any]] = []
    best_by_split_metric: dict[str, dict[str, dict[str, Any]]] = {}
    for split_name in COMPARISON_SPLITS:
        split_best: dict[str, dict[str, Any]] = {}
        for metric_name in COMPARISON_METRICS:
            split_best[metric_name] = {"run_name": None, "value": None}
        best_by_split_metric[split_name] = split_best

    for run in runs:
        row: dict[str, Any] = {
            "run_name": run.run_name,
            "run_dir": str(run.run_dir),
            "aggregate_metrics": run.aggregate_metrics,
            "splits": {},
        }
        for split_name in COMPARISON_SPLITS:
            metrics = run.per_split_metrics.get(split_name, {})
            per_field = metrics.get("per_field_accuracy", {})
            split_row = {
                "text_loss": float(metrics.get("text_loss", 0.0)),
                "exact_state_accuracy": float(metrics.get("exact_state_accuracy", 0.0)),
                "per_field_accuracy": {
                    "holder_accuracy": float(per_field.get("holder_accuracy", 0.0)),
                    "visibility_accuracy": float(per_field.get("visibility_accuracy", 0.0)),
                    "container_accuracy": float(per_field.get("container_accuracy", 0.0)),
                    "macro_accuracy": float(per_field.get("macro_accuracy", 0.0)),
                },
                "paraphrase_delta_cosine": float(metrics.get("paraphrase_delta_cosine", 0.0)),
                "example_count": float(metrics.get("example_count", 0.0)),
            }
            row["splits"][split_name] = split_row

            metric_values = {
                "text_loss": split_row["text_loss"],
                "exact_state_accuracy": split_row["exact_state_accuracy"],
                "per_field_accuracy": split_row["per_field_accuracy"]["macro_accuracy"],
                "paraphrase_delta_cosine": split_row["paraphrase_delta_cosine"],
            }
            for metric_name, value in metric_values.items():
                current_best = best_by_split_metric[split_name][metric_name]
                if current_best["value"] is None:
                    current_best["run_name"] = run.run_name
                    current_best["value"] = value
                    continue
                if metric_name == "text_loss":
                    better = value < current_best["value"]
                else:
                    better = value > current_best["value"]
                if better:
                    current_best["run_name"] = run.run_name
                    current_best["value"] = value
        comparison_rows.append(row)

    return {
        "splits": list(COMPARISON_SPLITS),
        "metrics": list(COMPARISON_METRICS),
        "runs": comparison_rows,
        "best_by_split_metric": best_by_split_metric,
    }


def render_comparison_markdown(payload: dict[str, Any]) -> str:
    """Render a stable markdown comparison table."""

    lines = [
        "# Benchmark Comparison",
        "",
    ]
    for split_name in payload["splits"]:
        lines.extend(
            [
                f"## {split_name}",
                "",
                "| run | text_loss | exact_state_accuracy | per_field_accuracy | paraphrase_delta_cosine | example_count |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        runs = sorted(payload["runs"], key=lambda row: row["run_name"])
        for run in runs:
            split_metrics = run["splits"][split_name]
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(run["run_name"]),
                        f"{float(split_metrics['text_loss']):.4f}",
                        f"{float(split_metrics['exact_state_accuracy']):.4f}",
                        f"{float(split_metrics['per_field_accuracy']['macro_accuracy']):.4f}",
                        f"{float(split_metrics['paraphrase_delta_cosine']):.4f}",
                        f"{float(split_metrics['example_count']):.0f}",
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_comparison_outputs(
    run_dirs: list[Path],
    *,
    out_dir: Path,
) -> dict[str, Any]:
    """Write combined comparison JSON and markdown for multiple runs."""

    payload = build_comparison_payload(run_dirs)
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = out_dir / "aggregate_metrics.json"
    summary_path = out_dir / "summary.md"
    aggregate_path.write_text(stable_json_dumps(payload) + "\n", encoding="utf-8")
    summary_path.write_text(render_comparison_markdown(payload), encoding="utf-8", newline="\n")
    return {
        "aggregate_path": str(aggregate_path),
        "summary_path": str(summary_path),
        "run_count": len(payload["runs"]),
    }
