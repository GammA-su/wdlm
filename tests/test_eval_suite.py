from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from wdlm.train.metrics import exact_state_accuracy, per_field_accuracy
from wdlm.train.trainer import evaluate_benchmark


pytestmark = pytest.mark.skipif(
    os.environ.get("WDLM_RUN_INTEGRATION_TESTS") != "1",
    reason="Benchmark evaluation integration test is opt-in. Set WDLM_RUN_INTEGRATION_TESTS=1 to enable.",
)


def _write_experiment_config(path: Path, *, split_dir: Path, output_dir: Path, model_type: str) -> None:
    path.write_text(
        "\n".join(
            [
                f"model_type: {model_type}",
                "seed: 42",
                "device: cpu",
                "precision: fp32",
                f"output_dir: {output_dir.as_posix()}",
                "best_metric: exact_state_accuracy",
                "data:",
                f"  train_path: {(split_dir / 'train.jsonl').as_posix()}",
                f"  val_path: {(split_dir / 'val.jsonl').as_posix()}",
                f"  split_dir: {split_dir.as_posix()}",
                "  benchmark_splits: [val, test_iid]",
                "  max_vocab_size: 64",
                "  max_seq_len: 24",
                "  max_paraphrases: 1",
                "model:",
                "  d_model: 16",
                "  n_heads: 2",
                "  n_layers: 1",
                "  ffn_dim: 32",
                "  dropout: 0.1",
                "  max_seq_len: 24",
                "  state_dim: 12",
                "optim:",
                "  batch_size: 2",
                "  lr: 0.001",
                "  epochs: 1",
                "  max_steps: 1",
                "loss:",
                "  text: 1.0",
                "  state: 1.0",
                "  paraphrase: 0.2",
                "  contrastive: 0.1",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_state_accuracy_metrics_are_correct() -> None:
    import torch

    from wdlm.train.dataset import StateTensorizer

    tensorizer = StateTensorizer(
        objects=["book"],
        holders=["alice", "table"],
        containers=["drawer"],
    )
    target = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]])
    perfect_scores = target.clone()
    wrong_scores = torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]])

    assert exact_state_accuracy(perfect_scores, target, state_tensorizer=tensorizer) == 1.0
    assert exact_state_accuracy(wrong_scores, target, state_tensorizer=tensorizer) == 0.0

    field_metrics = per_field_accuracy(wrong_scores, target, state_tensorizer=tensorizer)
    assert field_metrics["holder_accuracy"] == 0.0
    assert field_metrics["visibility_accuracy"] == 0.0
    assert field_metrics["container_accuracy"] == 0.0


def test_eval_suite_writes_reports(tmp_path: Path) -> None:
    split_dir = tmp_path / "splits"
    split_dir.mkdir(parents=True)
    config_path = tmp_path / "baseline_state.yaml"
    run_dir = tmp_path / "run"
    _write_experiment_config(
        config_path,
        split_dir=split_dir,
        output_dir=run_dir,
        model_type="state_head_baseline",
    )

    fake_metrics = {
        "text_loss": 1.0,
        "exact_state_accuracy": 0.5,
        "per_field_accuracy": {
            "holder_accuracy": 0.5,
            "visibility_accuracy": 0.5,
            "container_accuracy": 0.5,
            "macro_accuracy": 0.5,
        },
        "paraphrase_delta_cosine": 0.7,
        "example_count": 4.0,
        "device": "cpu",
    }
    with patch("wdlm.train.trainer.eval_model", side_effect=lambda *args, **kwargs: {**fake_metrics, "split": kwargs["split"]}):
        results = evaluate_benchmark(
            config_path,
            checkpoint_path=run_dir / "checkpoint_best.pt",
        )

    assert "aggregate_metrics" in results
    assert (run_dir / "benchmark_eval" / "metrics.json").exists()
    assert (run_dir / "benchmark_eval" / "per_split_metrics.json").exists()
    assert (run_dir / "benchmark_eval" / "summary.md").exists()

    aggregate_payload = json.loads((run_dir / "benchmark_eval" / "metrics.json").read_text(encoding="utf-8"))
    assert "aggregate_metrics" in aggregate_payload
