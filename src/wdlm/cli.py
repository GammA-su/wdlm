"""Command-line interface for WDLM data tools."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import typer

from wdlm.utils.io import read_jsonl, stable_json_dumps


app = typer.Typer(add_completion=False, no_args_is_help=True)


def _log(message: str) -> None:
    print(f"[wdlm.cli] {message}", flush=True)


def _emit_json(payload: Any) -> None:
    typer.echo(stable_json_dumps(payload))


def _count_jsonl(path: Path) -> int:
    return len(read_jsonl(path))


def _sample_step_summary(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    if not rows:
        return {"example_count": 0}
    first = rows[0]
    last = rows[-1]
    return {
        "example_count": len(rows),
        "first_example_id": first.get("example_id"),
        "last_example_id": last.get("example_id"),
        "first_world_id": first.get("world_id"),
        "first_action": first.get("action_struct"),
        "first_state_before": first.get("state_before"),
        "first_state_after": first.get("state_after"),
    }


def _trajectory_summary(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    if not rows:
        return {"trajectory_count": 0}
    difficulty_counts = Counter(str(row["metadata"]["difficulty"]) for row in rows)
    lengths = [int(row["metadata"]["episode_length"]) for row in rows]
    first = rows[0]
    return {
        "trajectory_count": len(rows),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "episode_length_min": min(lengths),
        "episode_length_max": max(lengths),
        "first_trajectory_id": first.get("trajectory_id"),
        "first_initial_state": first.get("initial_state"),
        "first_final_state": first.get("final_state"),
    }


def _qa_summary(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    if not rows:
        return {"qa_count": 0}
    question_counts = Counter(str(row["question_type"]) for row in rows)
    first = rows[0]
    return {
        "qa_count": len(rows),
        "question_type_counts": dict(sorted(question_counts.items())),
        "first_qa_id": first.get("qa_id"),
        "first_question": first.get("question"),
        "first_answer": first.get("answer"),
        "linked_example_id": first.get("example_id"),
    }


def _split_file_counts(paths: dict[str, Path]) -> dict[str, int]:
    return {split_name: _count_jsonl(path) for split_name, path in sorted(paths.items())}


def _log_config_summary(config_path: Path) -> None:
    from wdlm.train.trainer import load_train_config

    config = load_train_config(config_path)
    _log(
        "config="
        + stable_json_dumps(
            {
                "config_path": str(config_path),
                "model_type": config.model_type,
                "device": config.device,
                "output_dir": config.output_dir,
                "train_path": config.data.train_path,
                "val_path": config.data.val_path,
                "split_dir": config.data.split_dir,
                "batch_size": config.optim.batch_size,
                "epochs": config.optim.epochs,
                "max_steps": config.optim.max_steps,
                "lr": config.optim.lr,
                "state_dim": config.model.state_dim,
            }
        )
    )


def _progress_logger(prefix: str):
    def _callback(payload: dict[str, object]) -> None:
        _log(f"{prefix} progress={stable_json_dumps(payload)}")

    return _callback


@app.command("generate-toy-world")
def generate_toy_world_command(
    out: Path = typer.Option(..., help="Output JSONL path."),
    num_examples: int = typer.Option(..., min=1, help="Number of examples to create."),
    seed: int = typer.Option(42, help="Deterministic seed."),
    workers: int = typer.Option(0, min=0, help="Worker processes. 0 selects an automatic value."),
) -> None:
    """Generate a deterministic single-step toy-world JSONL dataset."""

    from wdlm.data.generate import generate_toy_world_file

    _log(f"generate-toy-world start out={out} num_examples={num_examples} seed={seed}")
    generate_toy_world_file(
        out=out,
        num_examples=num_examples,
        seed=seed,
        progress_callback=_progress_logger("generate-toy-world"),
        workers=workers,
    )
    summary = _sample_step_summary(out)
    _log(f"generate-toy-world wrote file={out} example_count={summary['example_count']}")
    _log(f"generate-toy-world sample_action={stable_json_dumps(summary.get('first_action'))}")
    _log(f"generate-toy-world sample_state_before={stable_json_dumps(summary.get('first_state_before'))}")
    _log(f"generate-toy-world sample_state_after={stable_json_dumps(summary.get('first_state_after'))}")
    typer.echo(f"Wrote {num_examples} examples to {out}")


@app.command("generate-trajectories")
def generate_trajectories_command(
    out_steps: Path = typer.Option(..., help="Output JSONL path for per-step examples."),
    out_trajectories: Path = typer.Option(..., help="Output JSONL path for trajectories."),
    num_trajectories: int = typer.Option(..., min=1, help="Number of trajectories to create."),
    seed: int = typer.Option(42, help="Deterministic seed."),
    workers: int = typer.Option(0, min=0, help="Worker processes. 0 selects an automatic value."),
) -> None:
    """Generate deterministic per-step and full-trajectory datasets."""

    from wdlm.data.trajectory import generate_trajectory_files

    _log(
        f"generate-trajectories start out_steps={out_steps} out_trajectories={out_trajectories} "
        f"num_trajectories={num_trajectories} seed={seed}"
    )
    step_path, trajectory_path = generate_trajectory_files(
        out_steps=out_steps,
        out_trajectories=out_trajectories,
        num_trajectories=num_trajectories,
        seed=seed,
        progress_callback=_progress_logger("generate-trajectories"),
        workers=workers,
    )
    step_summary = _sample_step_summary(step_path)
    trajectory_summary = _trajectory_summary(trajectory_path)
    _log(f"generate-trajectories step_summary={stable_json_dumps(step_summary)}")
    _log(f"generate-trajectories trajectory_summary={stable_json_dumps(trajectory_summary)}")
    typer.echo(f"steps: {step_path}")
    typer.echo(f"trajectories: {trajectory_path}")


@app.command("generate-qa")
def generate_qa_command(
    input_steps: Path = typer.Option(..., help="Input per-step JSONL file."),
    out: Path = typer.Option(..., help="Output QA JSONL path."),
    seed: int = typer.Option(42, help="Deterministic seed."),
) -> None:
    """Generate state-query QA examples from per-step records."""

    from wdlm.data.queries import generate_query_file

    _log(f"generate-qa start input_steps={input_steps} out={out} seed={seed}")
    _log(f"generate-qa input_step_count={_count_jsonl(input_steps)}")
    out_path = generate_query_file(input_steps_path=input_steps, out_path=out, seed=seed)
    _log(f"generate-qa summary={stable_json_dumps(_qa_summary(out_path))}")
    typer.echo(f"qa: {out_path}")


@app.command("split-dataset")
def split_dataset_command(
    input_path: Path = typer.Option(..., "--input", help="Input JSONL file."),
    out_dir: Path = typer.Option(..., help="Directory for split files."),
    seed: int = typer.Option(42, help="Deterministic split seed."),
    train_ratio: float = typer.Option(0.8, min=0.0, max=1.0, help="Train split ratio."),
    val_ratio: float = typer.Option(0.1, min=0.0, max=1.0, help="Validation split ratio."),
    test_ratio: float = typer.Option(0.1, min=0.0, max=1.0, help="Test split ratio."),
) -> None:
    """Split a JSONL dataset into train, validation, and test files."""

    from wdlm.data.split import split_dataset_file

    _log(
        f"split-dataset start input={input_path} out_dir={out_dir} seed={seed} "
        f"train_ratio={train_ratio} val_ratio={val_ratio} test_ratio={test_ratio}"
    )
    _log(f"split-dataset input_example_count={_count_jsonl(input_path)}")
    split_paths = split_dataset_file(
        input_path=input_path,
        out_dir=out_dir,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    _log(f"split-dataset output_counts={stable_json_dumps(_split_file_counts(split_paths))}")
    for split_name, path in split_paths.items():
        typer.echo(f"{split_name}: {path}")


@app.command("build-ood-splits")
def build_ood_splits_command(
    input_steps: Path = typer.Option(..., help="Input per-step JSONL file."),
    out_dir: Path = typer.Option(..., help="Directory for OOD split files."),
    seed: int = typer.Option(42, help="Deterministic split seed."),
    train_ratio: float = typer.Option(0.8, min=0.0, max=1.0, help="Train split ratio."),
    val_ratio: float = typer.Option(0.1, min=0.0, max=1.0, help="Validation split ratio."),
    test_ratio: float = typer.Option(0.1, min=0.0, max=1.0, help="IID test split ratio."),
) -> None:
    """Build IID and OOD split files from per-step examples."""

    from wdlm.data.ood import build_ood_splits_file

    _log(
        f"build-ood-splits start input_steps={input_steps} out_dir={out_dir} seed={seed} "
        f"train_ratio={train_ratio} val_ratio={val_ratio} test_ratio={test_ratio}"
    )
    _log(f"build-ood-splits input_step_count={_count_jsonl(input_steps)}")
    paths, summary = build_ood_splits_file(
        input_path=input_steps,
        out_dir=out_dir,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    _log(f"build-ood-splits output_counts={stable_json_dumps(_split_file_counts(paths))}")
    _log(f"build-ood-splits summary={stable_json_dumps(summary.__dict__)}")
    for split_name, path in paths.items():
        typer.echo(f"{split_name}: {path}")
    _emit_json(summary.__dict__)


@app.command("analyze-dataset")
def analyze_dataset_command(
    steps: Path = typer.Option(
        ...,
        "--steps",
        "--input",
        help="Per-step JSONL file to analyze. --steps is canonical; --input is a compatibility alias.",
    ),
    split_dir: Path | None = typer.Option(
        None,
        help="Optional split directory for OOD split statistics.",
    ),
) -> None:
    """Analyze a dataset and print aggregate statistics as JSON."""

    from wdlm.analysis.dataset_report import analyze_dataset_files

    _log(f"analyze-dataset start steps={steps} split_dir={split_dir}")
    _log(f"analyze-dataset step_count={_count_jsonl(steps)}")
    report = analyze_dataset_files(steps_path=steps, split_dir=split_dir)
    payload = report.model_dump(mode="json")
    _log(
        "analyze-dataset report_summary="
        + stable_json_dumps(
            {
                "example_count": payload.get("example_count"),
                "trajectory_count": payload.get("trajectory_count"),
                "avg_paraphrases_per_example": payload.get("avg_paraphrases_per_example"),
                "difficulty_counts": payload.get("difficulty_counts"),
            }
        )
    )
    _emit_json(payload)


@app.command("validate-dataset")
def validate_dataset_command(
    steps: Path = typer.Option(
        ...,
        "--steps",
        "--input",
        help="Per-step JSONL file to validate. --steps is canonical; --input is a compatibility alias.",
    ),
    trajectories: Path | None = typer.Option(
        None,
        help="Optional trajectory JSONL file to cross-check against steps.",
    ),
    qa: Path | None = typer.Option(
        None,
        help="Optional QA JSONL file to validate against steps.",
    ),
    split_dir: Path | None = typer.Option(
        None,
        help="Optional split directory to validate OOD membership and leakage.",
    ),
    seed: int = typer.Option(42, help="Deterministic seed used to build OOD splits."),
) -> None:
    """Validate WDLM Deliverable B artifacts and invariants."""

    from wdlm.analysis.validate_dataset import DatasetValidationError, validate_dataset_artifacts

    _log(
        f"validate-dataset start steps={steps} trajectories={trajectories} qa={qa} "
        f"split_dir={split_dir} seed={seed}"
    )
    _log(f"validate-dataset step_count={_count_jsonl(steps)}")
    if trajectories is not None and trajectories.exists():
        _log(f"validate-dataset trajectory_count={_count_jsonl(trajectories)}")
    if qa is not None and qa.exists():
        _log(f"validate-dataset qa_count={_count_jsonl(qa)}")
    try:
        report = validate_dataset_artifacts(
            steps_path=steps,
            trajectories_path=trajectories,
            qa_path=qa,
            split_dir=split_dir,
            seed=seed,
        )
    except DatasetValidationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    payload = report.model_dump(mode="json")
    _log(
        "validate-dataset summary="
        + stable_json_dumps(
            {
                "step_count": payload.get("step_count"),
                "trajectory_count": payload.get("trajectory_count"),
                "qa_count": payload.get("qa_count"),
                "split_files_checked": payload.get("split_file_count"),
            }
        )
    )
    _emit_json(payload)


@app.command("eval-exact-state")
def eval_exact_state_command(
    gold: Path = typer.Option(..., help="Gold JSONL file."),
    pred: Path = typer.Option(..., help="Predicted JSONL file."),
) -> None:
    """Evaluate exact state-after matches against gold JSONL."""

    from wdlm.eval.exact_state import evaluate_exact_state_files

    _log(f"eval-exact-state start gold={gold} pred={pred}")
    _log(f"eval-exact-state gold_count={_count_jsonl(gold)} pred_count={_count_jsonl(pred)}")
    metrics = evaluate_exact_state_files(gold_path=gold, pred_path=pred)
    payload = metrics.model_dump(mode="json")
    _log(
        "eval-exact-state summary="
        + stable_json_dumps(
            {
                "example_count": payload.get("example_count"),
                "exact_match_accuracy": payload.get("exact_match_accuracy"),
                "per_field_accuracy": payload.get("per_field_accuracy"),
            }
        )
    )
    _emit_json(payload)


@app.command("train-baseline")
def train_baseline_command(
    config: Path = typer.Option(..., help="YAML config path."),
) -> None:
    """Train the baseline language model."""

    _log(f"train-baseline start config={config}")
    _log("train-baseline importing trainer")
    from wdlm.train.trainer import load_train_config, train_model

    _log("train-baseline trainer imported")
    _log_config_summary(config)
    metrics = train_model(config, model_type="baseline")
    _log(
        "train-baseline done "
        + stable_json_dumps(
            {
                "output_dir": load_train_config(config).output_dir,
                "best_metric_name": metrics.get("best_metric_name"),
                "best_metric_value": metrics.get("best_metric_value"),
                "global_steps": metrics.get("global_steps"),
            }
        )
    )
    _emit_json(metrics)


@app.command("train-state-head-baseline")
def train_state_head_baseline_command(
    config: Path = typer.Option(..., help="YAML config path."),
) -> None:
    """Train the causal LM baseline with an auxiliary state head."""

    _log(f"train-state-head-baseline start config={config}")
    _log("train-state-head-baseline importing trainer")
    from wdlm.train.trainer import load_train_config, train_model

    _log("train-state-head-baseline trainer imported")
    _log_config_summary(config)
    metrics = train_model(config, model_type="state_head_baseline")
    _log(
        "train-state-head-baseline done "
        + stable_json_dumps(
            {
                "output_dir": load_train_config(config).output_dir,
                "best_metric_name": metrics.get("best_metric_name"),
                "best_metric_value": metrics.get("best_metric_value"),
                "global_steps": metrics.get("global_steps"),
            }
        )
    )
    _emit_json(metrics)


@app.command("train-para-align-baseline")
def train_para_align_baseline_command(
    config: Path = typer.Option(..., help="YAML config path."),
) -> None:
    """Train the causal LM baseline with paraphrase-aligned representations."""

    _log(f"train-para-align-baseline start config={config}")
    _log("train-para-align-baseline importing trainer")
    from wdlm.train.trainer import load_train_config, train_model

    _log("train-para-align-baseline trainer imported")
    _log_config_summary(config)
    metrics = train_model(config, model_type="para_align_baseline")
    _log(
        "train-para-align-baseline done "
        + stable_json_dumps(
            {
                "output_dir": load_train_config(config).output_dir,
                "best_metric_name": metrics.get("best_metric_name"),
                "best_metric_value": metrics.get("best_metric_value"),
                "global_steps": metrics.get("global_steps"),
            }
        )
    )
    _emit_json(metrics)


@app.command("train-wdlm")
def train_wdlm_command(
    config: Path = typer.Option(..., help="YAML config path."),
) -> None:
    """Train the minimal WDLM scaffold."""

    _log(f"train-wdlm start config={config}")
    _log("train-wdlm importing trainer")
    from wdlm.train.trainer import load_train_config, train_model

    _log("train-wdlm trainer imported")
    _log_config_summary(config)
    metrics = train_model(config, model_type="wdlm")
    _log(
        "train-wdlm done "
        + stable_json_dumps(
            {
                "output_dir": load_train_config(config).output_dir,
                "best_metric_name": metrics.get("best_metric_name"),
                "best_metric_value": metrics.get("best_metric_value"),
                "global_steps": metrics.get("global_steps"),
            }
        )
    )
    _emit_json(metrics)


@app.command("eval-model")
def eval_model_command(
    config: Path = typer.Option(..., help="YAML config path."),
    checkpoint: Path = typer.Option(..., help="Checkpoint path."),
    split: str = typer.Option("val", help="Dataset split to evaluate."),
) -> None:
    """Evaluate a saved training checkpoint."""

    _log(f"eval-model start config={config} checkpoint={checkpoint} split={split}")
    _log("eval-model importing trainer")
    from wdlm.train.trainer import eval_model

    _log("eval-model trainer imported")
    metrics = eval_model(config, checkpoint_path=checkpoint, split=split)
    _log(
        "eval-model summary="
        + stable_json_dumps(
            {
                "split": metrics.get("split"),
                "device": metrics.get("device"),
                "text_loss": metrics.get("text_loss"),
                "exact_state_accuracy": metrics.get("exact_state_accuracy"),
                "per_field_accuracy": metrics.get("per_field_accuracy"),
                "paraphrase_delta_cosine": metrics.get("paraphrase_delta_cosine"),
                "example_count": metrics.get("example_count"),
            }
        )
    )
    _emit_json(metrics)


@app.command("eval-benchmark")
def eval_benchmark_command(
    config: Path = typer.Option(..., help="YAML config path."),
    checkpoint: Path = typer.Option(..., help="Checkpoint path."),
    out_dir: Path | None = typer.Option(None, help="Optional output directory for benchmark reports."),
    splits: str | None = typer.Option(
        None,
        help="Comma-separated split names. Defaults to config.data.benchmark_splits.",
    ),
) -> None:
    """Evaluate a checkpoint across benchmark splits and write report files."""

    _log(f"eval-benchmark start config={config} checkpoint={checkpoint} out_dir={out_dir}")
    _log("eval-benchmark importing trainer")
    from wdlm.train.trainer import evaluate_benchmark

    split_names = None if splits is None else [item.strip() for item in splits.split(",") if item.strip()]
    _log("eval-benchmark trainer imported")
    _log(
        f"eval-benchmark resolved_splits={split_names if split_names is not None else 'config-default'}"
    )
    metrics = evaluate_benchmark(
        config,
        checkpoint_path=checkpoint,
        output_dir=out_dir,
        splits=split_names,
    )
    _log(
        "eval-benchmark summary="
        + stable_json_dumps(
            {
                "summary_path": metrics.get("summary_path"),
                "evaluated_splits": sorted(metrics.get("per_split_metrics", {}).keys()),
            }
        )
    )
    _emit_json(metrics)


@app.command("run-experiment")
def run_experiment_command(
    config: Path = typer.Option(..., help="YAML config path."),
) -> None:
    """Train a model and evaluate the best checkpoint on benchmark splits."""

    _log(f"run-experiment start config={config}")
    _log("run-experiment importing trainer")
    from wdlm.train.trainer import run_experiment

    _log("run-experiment trainer imported")
    _log_config_summary(config)
    metrics = run_experiment(config)
    benchmark = metrics.get("benchmark", {})
    _log(
        "run-experiment summary="
        + stable_json_dumps(
            {
                "summary_path": benchmark.get("summary_path"),
                "evaluated_splits": sorted(benchmark.get("per_split_metrics", {}).keys()),
            }
        )
    )
    _emit_json(metrics)


@app.command("compare-benchmark-runs")
def compare_benchmark_runs_command(
    runs: str = typer.Option(..., help="Comma-separated run directories to compare."),
    out_dir: Path = typer.Option(..., help="Directory for aggregate comparison outputs."),
    memo_out: Path | None = typer.Option(
        None,
        help="Optional path for docs/first_result_memo.md-style output.",
    ),
) -> None:
    """Aggregate benchmark results from multiple run directories."""

    from wdlm.analysis.comparison import write_comparison_outputs
    from wdlm.analysis.memo import write_first_result_memo

    run_dirs = [Path(item.strip()) for item in runs.split(",") if item.strip()]
    _log(f"compare-benchmark-runs start run_dirs={[str(path) for path in run_dirs]} out_dir={out_dir} memo_out={memo_out}")
    result = write_comparison_outputs(run_dirs, out_dir=out_dir)
    if memo_out is not None:
        import json

        payload = json.loads(Path(result["aggregate_path"]).read_text(encoding="utf-8"))
        write_first_result_memo(payload, out_path=memo_out)
        _log(f"compare-benchmark-runs wrote memo={memo_out}")
    _log(
        "compare-benchmark-runs summary="
        + stable_json_dumps(
            {
                "aggregate_path": result.get("aggregate_path"),
                "summary_path": result.get("summary_path"),
                "run_count": result.get("run_count"),
            }
        )
    )
    _emit_json(result)


@app.command("run-ablation-suite")
def run_ablation_suite_command(
    config_dir: Path = typer.Option(Path("configs"), help="Directory containing ablation config files."),
    out_dir: Path = typer.Option(Path("runs/ablation_suite"), help="Directory for suite aggregate outputs."),
    include_core_models: bool = typer.Option(
        False,
        "--include-core-models/--wdlm-only",
        help="Include the three non-WDLM core baselines alongside the WDLM ablation set.",
    ),
    memo_out: Path = typer.Option(
        Path("docs/first_result_memo.md"),
        help="Where to write the first result memo scaffold.",
    ),
) -> None:
    """Run the WDLM ablation suite and write aggregate comparison outputs."""

    from wdlm.train.suites import run_ablation_suite

    _log(
        f"run-ablation-suite start config_dir={config_dir} out_dir={out_dir} "
        f"include_core_models={include_core_models} memo_out={memo_out}"
    )
    result = run_ablation_suite(
        config_dir=config_dir,
        out_dir=out_dir,
        include_core_models=include_core_models,
        memo_out=memo_out,
    )
    _log(
        "run-ablation-suite summary="
        + stable_json_dumps(
            {
                "aggregate_path": result.get("comparison", {}).get("aggregate_path"),
                "summary_path": result.get("comparison", {}).get("summary_path"),
                "memo_path": result.get("memo_path"),
                "run_count": len(result.get("run_dirs", [])),
            }
        )
    )
    _emit_json(result)


def main() -> None:
    """Run the CLI."""

    app()


if __name__ == "__main__":
    main()
