"""Experiment suite runners for WDLM toy-world comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from wdlm.analysis.comparison import write_comparison_outputs
from wdlm.analysis.memo import write_first_result_memo
from wdlm.train.trainer import load_train_config, run_experiment
from wdlm.utils.io import stable_json_dumps


WDLM_ABLATION_CONFIGS: tuple[str, ...] = (
    "toy_wdlm.yaml",
    "toy_wdlm_no_para.yaml",
    "toy_wdlm_no_state.yaml",
    "toy_wdlm_no_contrastive.yaml",
    "toy_wdlm_no_conditioning.yaml",
    "toy_wdlm_state128.yaml",
    "toy_wdlm_state512.yaml",
)

CORE_MODEL_CONFIGS: tuple[str, ...] = (
    "toy_baseline.yaml",
    "toy_state_head_baseline.yaml",
    "toy_para_align_baseline.yaml",
    "toy_wdlm.yaml",
)


def _resolve_config_paths(config_names: tuple[str, ...], config_dir: Path) -> list[Path]:
    paths = [config_dir / name for name in config_names]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing config files: {missing}")
    return paths


def run_ablation_suite(
    *,
    config_dir: Path,
    out_dir: Path,
    include_core_models: bool = False,
    memo_out: Path | None = None,
) -> dict[str, Any]:
    """Run the WDLM ablation suite and aggregate benchmark outputs."""

    config_paths = _resolve_config_paths(WDLM_ABLATION_CONFIGS, config_dir)
    if include_core_models:
        core_paths = _resolve_config_paths(CORE_MODEL_CONFIGS[:-1], config_dir)
        config_paths = core_paths + config_paths

    run_dirs: list[Path] = []
    experiment_reports: dict[str, Any] = {}
    for config_path in config_paths:
        report = run_experiment(config_path)
        config = load_train_config(config_path)
        run_dir = Path(config.output_dir)
        run_dirs.append(run_dir)
        experiment_reports[run_dir.name] = report

    comparison = write_comparison_outputs(run_dirs, out_dir=out_dir)
    payload_path = Path(comparison["aggregate_path"])
    import json

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    memo_path = write_first_result_memo(
        payload,
        out_path=memo_out or Path("docs/first_result_memo.md"),
    )
    report = {
        "config_paths": [str(path) for path in config_paths],
        "run_dirs": [str(path) for path in run_dirs],
        "comparison": comparison,
        "memo_path": str(memo_path),
        "experiments": experiment_reports,
    }
    report_path = out_dir / "suite_report.json"
    report_path.write_text(stable_json_dumps(report) + "\n", encoding="utf-8", newline="\n")
    return report
