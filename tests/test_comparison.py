from __future__ import annotations

import json
from pathlib import Path

from wdlm.analysis.comparison import build_comparison_payload, render_comparison_markdown, write_comparison_outputs
from wdlm.analysis.memo import write_first_result_memo
from wdlm.utils.io import stable_json_dumps


def _benchmark_payload(*, base_exact: float, base_loss: float) -> tuple[dict[str, object], dict[str, object]]:
    per_split = {}
    for index, split_name in enumerate(
        [
            "val",
            "test_iid",
            "test_lexical_ood",
            "test_paraphrase_ood",
            "test_compositional_ood",
            "test_length_ood",
        ]
    ):
        per_split[split_name] = {
            "split": split_name,
            "text_loss": base_loss + index * 0.1,
            "exact_state_accuracy": max(base_exact - index * 0.02, 0.0),
            "per_field_accuracy": {
                "holder_accuracy": max(base_exact - index * 0.02, 0.0),
                "visibility_accuracy": max(base_exact - index * 0.01, 0.0),
                "container_accuracy": max(base_exact - index * 0.03, 0.0),
                "macro_accuracy": max(base_exact - index * 0.02, 0.0),
            },
            "paraphrase_delta_cosine": max(0.7 - index * 0.03, 0.0),
            "example_count": 10.0 + index,
        }
    metrics = {
        "aggregate_metrics": {
            "text_loss": base_loss,
            "exact_state_accuracy": base_exact,
        }
    }
    return metrics, per_split


def _write_mock_run(run_dir: Path, *, base_exact: float, base_loss: float) -> None:
    benchmark_dir = run_dir / "benchmark_eval"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    metrics, per_split = _benchmark_payload(base_exact=base_exact, base_loss=base_loss)
    (benchmark_dir / "metrics.json").write_text(stable_json_dumps(metrics) + "\n", encoding="utf-8")
    (benchmark_dir / "per_split_metrics.json").write_text(stable_json_dumps(per_split) + "\n", encoding="utf-8")


def test_comparison_aggregator_and_memo_outputs(tmp_path: Path) -> None:
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    _write_mock_run(run_a, base_exact=0.70, base_loss=1.1)
    _write_mock_run(run_b, base_exact=0.75, base_loss=1.0)

    payload = build_comparison_payload([run_b, run_a])
    assert [run["run_name"] for run in payload["runs"]] == ["run_a", "run_b"]
    assert payload["best_by_split_metric"]["val"]["exact_state_accuracy"]["run_name"] == "run_b"

    markdown = render_comparison_markdown(payload)
    assert markdown.startswith("# Benchmark Comparison\n")
    assert "| run_a | 1.1000 | 0.7000 | 0.7000 | 0.7000 | 10 |" in markdown
    assert "| run_b | 1.0000 | 0.7500 | 0.7500 | 0.7000 | 10 |" in markdown

    out_dir = tmp_path / "comparison"
    outputs = write_comparison_outputs([run_a, run_b], out_dir=out_dir)
    assert Path(outputs["aggregate_path"]).exists()
    assert Path(outputs["summary_path"]).exists()

    memo_path = tmp_path / "docs" / "first_result_memo.md"
    memo_written = write_first_result_memo(payload, out_path=memo_path)
    memo_text = memo_written.read_text(encoding="utf-8")
    assert "## Experiment List" in memo_text
    assert "`run_b`" in memo_text


def test_markdown_summary_is_stable(tmp_path: Path) -> None:
    run_a = tmp_path / "alpha"
    run_b = tmp_path / "beta"
    _write_mock_run(run_a, base_exact=0.65, base_loss=1.2)
    _write_mock_run(run_b, base_exact=0.80, base_loss=0.9)

    payload = build_comparison_payload([run_a, run_b])
    markdown = render_comparison_markdown(payload)
    lines = markdown.splitlines()
    assert lines[0] == "# Benchmark Comparison"
    assert lines[2] == "## val"
    assert lines[3] == ""
    assert lines[4] == "| run | text_loss | exact_state_accuracy | per_field_accuracy | paraphrase_delta_cosine | example_count |"
    assert lines[5] == "| --- | ---: | ---: | ---: | ---: | ---: |"
    assert lines[6] == "| alpha | 1.2000 | 0.6500 | 0.6500 | 0.7000 | 10 |"
    assert lines[7] == "| beta | 0.9000 | 0.8000 | 0.8000 | 0.7000 | 10 |"
