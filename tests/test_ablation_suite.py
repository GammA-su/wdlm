from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from wdlm.train.suites import WDLM_ABLATION_CONFIGS, run_ablation_suite
from wdlm.utils.io import stable_json_dumps


def _write_minimal_config(path: Path, *, output_dir: Path) -> None:
    path.write_text(
        f"model_type: wdlm\nseed: 42\ndevice: cpu\nprecision: fp32\noutput_dir: {output_dir.as_posix()}\n"
        "data:\n  train_path: data/train.jsonl\n  val_path: data/val.jsonl\n"
        "model:\n  d_model: 32\n  n_heads: 4\n  n_layers: 2\n  ffn_dim: 64\n  dropout: 0.1\n  max_seq_len: 32\n  state_dim: 24\n"
        "optim:\n  batch_size: 4\n  lr: 0.001\n  epochs: 1\n"
        "loss:\n  text: 1.0\n  state: 1.0\n  paraphrase: 0.2\n  contrastive: 0.1\n",
        encoding="utf-8",
    )


def _write_mock_benchmark(run_dir: Path, *, exact_value: float) -> None:
    benchmark_dir = run_dir / "benchmark_eval"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    per_split = {
        split_name: {
            "split": split_name,
            "text_loss": 1.0,
            "exact_state_accuracy": exact_value,
            "per_field_accuracy": {
                "holder_accuracy": exact_value,
                "visibility_accuracy": exact_value,
                "container_accuracy": exact_value,
                "macro_accuracy": exact_value,
            },
            "paraphrase_delta_cosine": 0.6,
            "example_count": 5.0,
        }
        for split_name in [
            "val",
            "test_iid",
            "test_lexical_ood",
            "test_paraphrase_ood",
            "test_compositional_ood",
            "test_length_ood",
        ]
    }
    metrics = {"aggregate_metrics": {"exact_state_accuracy": exact_value, "text_loss": 1.0}}
    (benchmark_dir / "per_split_metrics.json").write_text(stable_json_dumps(per_split) + "\n", encoding="utf-8")
    (benchmark_dir / "metrics.json").write_text(stable_json_dumps(metrics) + "\n", encoding="utf-8")


def test_ablation_runner_produces_summary_files(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    output_dirs: dict[str, Path] = {}
    for index, config_name in enumerate(WDLM_ABLATION_CONFIGS):
        output_dir = tmp_path / "runs" / Path(config_name).stem
        output_dirs[config_name] = output_dir
        _write_minimal_config(config_dir / config_name, output_dir=output_dir)

    def fake_run_experiment(config_path: Path) -> dict[str, object]:
        run_dir = output_dirs[config_path.name]
        _write_mock_benchmark(run_dir, exact_value=0.5 + 0.01 * len(config_path.name))
        return {"ok": True, "config": config_path.name}

    def fake_load_train_config(config_path: Path) -> SimpleNamespace:
        return SimpleNamespace(output_dir=str(output_dirs[config_path.name]))

    monkeypatch.setattr("wdlm.train.suites.run_experiment", fake_run_experiment)
    monkeypatch.setattr("wdlm.train.suites.load_train_config", fake_load_train_config)

    out_dir = tmp_path / "suite"
    memo_out = tmp_path / "docs" / "first_result_memo.md"
    result = run_ablation_suite(
        config_dir=config_dir,
        out_dir=out_dir,
        include_core_models=False,
        memo_out=memo_out,
    )

    assert Path(result["comparison"]["aggregate_path"]).exists()
    assert Path(result["comparison"]["summary_path"]).exists()
    assert (out_dir / "suite_report.json").exists()
    assert memo_out.exists()
