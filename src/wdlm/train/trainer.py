"""Training and evaluation utilities for WDLM toy-world experiments."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader

from wdlm.models import (
    BaselineLanguageModel,
    ParaphraseAlignedBaselineModel,
    StateHeadBaselineModel,
    WDLMModel,
)
from wdlm.train.collate import collate_examples
from wdlm.train.dataset import SimpleTokenizer, StateTensorizer, StepDataset
from wdlm.train.losses import (
    causal_text_cross_entropy,
    paraphrase_delta_invariance_loss,
    state_mse_loss,
    supervised_contrastive_delta_loss,
)
from wdlm.train.metrics import (
    delta_norm_stats,
    exact_state_accuracy,
    flatten_metric_dict,
    paraphrase_delta_cosine,
    per_field_accuracy,
)
from wdlm.utils.io import ensure_parent_dir, stable_json_dumps


ModelType = Literal["baseline", "state_head_baseline", "para_align_baseline", "wdlm"]
DeviceType = Literal["auto", "cpu", "cuda"]
PrecisionType = Literal["auto", "fp32", "bf16", "fp16"]


DEFAULT_BENCHMARK_SPLITS: tuple[str, ...] = (
    "val",
    "test_iid",
    "test_lexical_ood",
    "test_paraphrase_ood",
    "test_compositional_ood",
    "test_length_ood",
)


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_path: str
    val_path: str | None = None
    split_dir: str | None = None
    split_paths: dict[str, str] = Field(default_factory=dict)
    benchmark_splits: list[str] = Field(default_factory=lambda: list(DEFAULT_BENCHMARK_SPLITS))
    max_vocab_size: int = 512
    max_seq_len: int = 64
    max_paraphrases: int = 3
    num_workers: int | None = None
    pin_memory: bool | None = None
    persistent_workers: bool | None = None
    prefetch_factor: int | None = 2


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    ffn_dim: int = 128
    dropout: float = 0.1
    max_seq_len: int = 64
    state_dim: int = 64
    use_state_conditioning: bool = True


class OptimConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: int = 8
    lr: float = 1e-3
    epochs: int = 1
    max_steps: int | None = None
    resume_from: str | None = None


class LossConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: float = 1.0
    state: float = 1.0
    paraphrase: float = 0.2
    contrastive: float = 0.1


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_type: ModelType | None = None
    seed: int = 42
    device: DeviceType = "auto"
    precision: PrecisionType = "auto"
    output_dir: str
    best_metric: str = "per_field_macro_accuracy"
    best_metric_mode: Literal["min", "max"] | None = None
    data: DataConfig
    model: ModelConfig
    optim: OptimConfig
    loss: LossConfig


def load_train_config(path: Path) -> TrainConfig:
    """Load a YAML training config."""

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return TrainConfig.model_validate(payload)


def set_global_seed(seed: int) -> None:
    """Set deterministic seeds for CPU-friendly training."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def configure_runtime_for_speed(device: torch.device) -> None:
    """Enable safe runtime speed settings for the current device."""

    cpu_threads = min(16, os.process_cpu_count() or os.cpu_count() or 1)
    torch.set_num_threads(cpu_threads)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def resolve_device(requested: DeviceType) -> torch.device:
    """Resolve a training/evaluation device from config."""

    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Config requested device='cuda', but CUDA is not available.")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_precision(requested: PrecisionType, device: torch.device) -> torch.dtype | None:
    """Resolve autocast precision for the active device."""

    if device.type != "cuda":
        return None
    if requested == "fp32":
        return None
    if requested == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("Config requested precision='bf16', but CUDA bf16 is not supported.")
        return torch.bfloat16
    if requested == "fp16":
        return torch.float16
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _resolve_best_metric_mode(metric_name: str, explicit_mode: str | None) -> Literal["min", "max"]:
    if explicit_mode in {"min", "max"}:
        return explicit_mode
    lowered = metric_name.lower()
    if "loss" in lowered:
        return "min"
    return "max"


def _metric_better(current: float, best: float | None, mode: Literal["min", "max"]) -> bool:
    if best is None:
        return True
    return current < best if mode == "min" else current > best


def _extract_metric_value(metrics: dict[str, Any], metric_name: str) -> float:
    if metric_name == "per_field_macro_accuracy":
        return float(metrics.get("per_field_accuracy", {}).get("macro_accuracy", 0.0))
    current: Any = metrics
    for part in metric_name.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Metric '{metric_name}' not found in metrics: {sorted(metrics)}")
        current = current[part]
    if isinstance(current, dict):
        if "macro_accuracy" in current:
            return float(current["macro_accuracy"])
        raise TypeError(f"Metric '{metric_name}' resolved to a dict without macro_accuracy.")
    return float(current)


def _macro_per_field_accuracy(metrics: dict[str, Any]) -> float:
    return float(metrics.get("per_field_accuracy", {}).get("macro_accuracy", 0.0))


def _partial_learning_note(metrics: dict[str, Any], previous_macro: float | None = None) -> str | None:
    exact = float(metrics.get("exact_state_accuracy", 0.0))
    macro = _macro_per_field_accuracy(metrics)
    if exact != 0.0 or macro <= 0.0:
        return None
    if previous_macro is not None and macro <= previous_macro:
        return None
    return (
        "[wdlm] note: exact_state_accuracy is a strict all-fields metric. "
        f"Partial learning exists: per_field_macro_accuracy={macro:.4f} while exact_state_accuracy remains 0.0000."
    )


def _resolve_split_path(config: TrainConfig, split: str) -> Path:
    if split == "train":
        return Path(config.data.train_path)
    if split == "val":
        return Path(config.data.val_path or config.data.train_path)
    if split in config.data.split_paths:
        return Path(config.data.split_paths[split])
    if config.data.split_dir is not None:
        return Path(config.data.split_dir) / f"{split}.jsonl"
    raise ValueError(f"Unable to resolve split path for split '{split}'.")


def _build_dataset(
    config: TrainConfig,
    *,
    split: str,
    tokenizer: SimpleTokenizer | None = None,
    state_tensorizer: StateTensorizer | None = None,
) -> StepDataset:
    return StepDataset(
        _resolve_split_path(config, split),
        tokenizer=tokenizer,
        state_tensorizer=state_tensorizer,
        max_vocab_size=config.data.max_vocab_size,
        max_seq_len=config.data.max_seq_len,
    )


def _make_dataloader(
    dataset: StepDataset,
    config: TrainConfig,
    *,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    num_workers = 0 if config.data.num_workers is None else max(0, config.data.num_workers)
    pin_memory = bool(config.data.pin_memory) if config.data.pin_memory is not None else device.type == "cuda"
    persistent_workers = (
        bool(config.data.persistent_workers)
        if config.data.persistent_workers is not None
        else num_workers > 0
    )
    dataloader_kwargs: dict[str, object] = {
        "batch_size": config.optim.batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers if num_workers > 0 else False,
        "collate_fn": lambda samples: collate_examples(
            samples,
            pad_id=dataset.tokenizer.pad_id,
            max_paraphrases=config.data.max_paraphrases,
        ),
    }
    if num_workers > 0 and config.data.prefetch_factor is not None:
        dataloader_kwargs["prefetch_factor"] = config.data.prefetch_factor
    return DataLoader(
        dataset,
        **dataloader_kwargs,
    )


def _build_model(
    model_type: ModelType,
    *,
    vocab_size: int,
    state_input_dim: int,
    model_config: ModelConfig,
) -> torch.nn.Module:
    if model_type == "baseline":
        return BaselineLanguageModel(
            vocab_size=vocab_size,
            d_model=model_config.d_model,
            n_heads=model_config.n_heads,
            n_layers=model_config.n_layers,
            ffn_dim=model_config.ffn_dim,
            max_seq_len=model_config.max_seq_len,
            dropout=model_config.dropout,
        )
    if model_type == "state_head_baseline":
        return StateHeadBaselineModel(
            vocab_size=vocab_size,
            state_input_dim=state_input_dim,
            d_model=model_config.d_model,
            n_heads=model_config.n_heads,
            n_layers=model_config.n_layers,
            ffn_dim=model_config.ffn_dim,
            max_seq_len=model_config.max_seq_len,
            dropout=model_config.dropout,
            state_dim=model_config.state_dim,
        )
    if model_type == "para_align_baseline":
        return ParaphraseAlignedBaselineModel(
            vocab_size=vocab_size,
            state_input_dim=state_input_dim,
            d_model=model_config.d_model,
            n_heads=model_config.n_heads,
            n_layers=model_config.n_layers,
            ffn_dim=model_config.ffn_dim,
            max_seq_len=model_config.max_seq_len,
            dropout=model_config.dropout,
            state_dim=model_config.state_dim,
        )
    return WDLMModel(
        vocab_size=vocab_size,
        state_input_dim=state_input_dim,
        d_model=model_config.d_model,
        n_heads=model_config.n_heads,
        n_layers=model_config.n_layers,
        ffn_dim=model_config.ffn_dim,
        max_seq_len=model_config.max_seq_len,
        dropout=model_config.dropout,
        state_dim=model_config.state_dim,
        use_state_conditioning=model_config.use_state_conditioning,
    )


def _move_batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    moved: dict[str, object] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=device.type == "cuda") if isinstance(value, Tensor) else value
    return moved


def _forward_with_optional_state(
    model_type: ModelType,
    model: torch.nn.Module,
    *,
    input_ids: Tensor,
    attention_mask: Tensor,
    state_before: Tensor,
) -> dict[str, Tensor]:
    if model_type == "baseline":
        return model(input_ids=input_ids, attention_mask=attention_mask)  # type: ignore[operator]
    return model(  # type: ignore[operator]
        input_ids=input_ids,
        attention_mask=attention_mask,
        state_before=state_before,
    )


def _compute_auxiliary_outputs(
    model_type: ModelType,
    model: torch.nn.Module,
    *,
    input_ids: Tensor,
    attention_mask: Tensor,
    state_before: Tensor,
) -> dict[str, Tensor]:
    if input_ids.size(0) == 0:
        feature_dim = model.backbone.token_embedding.embedding_dim  # type: ignore[attr-defined]
        hidden_dim = getattr(model, "state_encoder", None)
        if hidden_dim is not None:
            feature_dim = model.state_encoder.net[-1].out_features  # type: ignore[attr-defined]
        return {
            "delta_features": state_before.new_zeros((0, feature_dim)),
        }
    return _forward_with_optional_state(
        model_type,
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        state_before=state_before,
    )


def _compute_state_prediction_tensor(
    model_type: ModelType,
    outputs: dict[str, Tensor],
    *,
    state_before: Tensor,
) -> Tensor:
    if "state_logits" in outputs:
        return outputs["state_logits"]
    if model_type == "baseline":
        return state_before
    raise KeyError(f"Missing state prediction path for model_type={model_type}.")


def _compute_batch_metrics(
    model_type: ModelType,
    outputs: dict[str, Tensor],
    *,
    batch: dict[str, object],
    state_tensorizer: StateTensorizer,
    paraphrase_outputs: dict[str, Tensor] | None,
) -> dict[str, float | dict[str, float]]:
    state_before = batch["state_before"]
    state_after = batch["state_after"]
    if not isinstance(state_before, Tensor) or not isinstance(state_after, Tensor):
        raise TypeError("Missing explicit state tensors in batch.")

    state_prediction = _compute_state_prediction_tensor(
        model_type,
        outputs,
        state_before=state_before,
    )
    field_metrics = per_field_accuracy(
        state_prediction,
        state_after,
        state_tensorizer=state_tensorizer,
    )
    metrics: dict[str, float | dict[str, float]] = {
        "example_count": float(state_after.size(0)),
        "exact_state_accuracy": exact_state_accuracy(
            state_prediction,
            state_after,
            state_tensorizer=state_tensorizer,
        ),
        "per_field_accuracy": field_metrics,
    }
    if paraphrase_outputs is not None:
        owner_indices = batch["paraphrase_owner_indices"]
        if not isinstance(owner_indices, Tensor):
            raise TypeError("Missing paraphrase_owner_indices in batch.")
        metrics["paraphrase_delta_cosine"] = paraphrase_delta_cosine(
            outputs["delta_features"],
            paraphrase_outputs["delta_features"],
            owner_indices,
        )
    else:
        metrics["paraphrase_delta_cosine"] = 0.0
    metrics.update(delta_norm_stats(outputs["delta_features"]))
    return metrics


def _compute_losses(
    model_type: ModelType,
    model: torch.nn.Module,
    batch: dict[str, object],
    *,
    pad_id: int,
    loss_config: LossConfig,
    state_tensorizer: StateTensorizer,
) -> dict[str, Tensor | float | dict[str, float]]:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    state_before = batch["state_before"]
    state_after = batch["state_after"]
    if (
        not isinstance(input_ids, Tensor)
        or not isinstance(attention_mask, Tensor)
        or not isinstance(labels, Tensor)
        or not isinstance(state_before, Tensor)
        or not isinstance(state_after, Tensor)
    ):
        raise TypeError("Missing required tensors in batch.")

    outputs = _forward_with_optional_state(
        model_type,
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        state_before=state_before,
    )
    text_loss = causal_text_cross_entropy(outputs["logits"], labels, pad_id=pad_id)

    if "state_logits" in outputs:
        explicit_state_loss = state_mse_loss(torch.sigmoid(outputs["state_logits"]), state_after)
    else:
        explicit_state_loss = text_loss.new_zeros(())
    if model_type == "wdlm":
        with torch.no_grad():
            latent_target = model.encode_state(state_after)  # type: ignore[attr-defined]
        latent_state_loss = state_mse_loss(outputs["s_pred"], latent_target)
        state_loss = latent_state_loss + explicit_state_loss
    else:
        state_loss = explicit_state_loss

    paraphrase_input_ids = batch["paraphrase_input_ids"]
    paraphrase_attention_mask = batch["paraphrase_attention_mask"]
    paraphrase_owner_indices = batch["paraphrase_owner_indices"]
    if (
        not isinstance(paraphrase_input_ids, Tensor)
        or not isinstance(paraphrase_attention_mask, Tensor)
        or not isinstance(paraphrase_owner_indices, Tensor)
    ):
        raise TypeError("Missing paraphrase tensors in batch.")
    if paraphrase_input_ids.size(0) > 0:
        paraphrase_state_before = state_before.index_select(0, paraphrase_owner_indices)
        paraphrase_outputs = _compute_auxiliary_outputs(
            model_type,
            model,
            input_ids=paraphrase_input_ids,
            attention_mask=paraphrase_attention_mask,
            state_before=paraphrase_state_before,
        )
        paraphrase_loss = paraphrase_delta_invariance_loss(
            outputs["delta_features"],
            paraphrase_outputs["delta_features"],
            paraphrase_owner_indices,
        )
    else:
        paraphrase_outputs = None
        paraphrase_loss = text_loss.new_zeros(())

    negative_input_ids = batch["negative_input_ids"]
    negative_attention_mask = batch["negative_attention_mask"]
    negative_owner_indices = batch["negative_owner_indices"]
    if (
        not isinstance(negative_input_ids, Tensor)
        or not isinstance(negative_attention_mask, Tensor)
        or not isinstance(negative_owner_indices, Tensor)
    ):
        raise TypeError("Missing negative tensors in batch.")
    if negative_input_ids.size(0) > 0:
        negative_state_before = state_before.index_select(0, negative_owner_indices)
        negative_outputs = _compute_auxiliary_outputs(
            model_type,
            model,
            input_ids=negative_input_ids,
            attention_mask=negative_attention_mask,
            state_before=negative_state_before,
        )
        contrastive_loss = supervised_contrastive_delta_loss(
            outputs["delta_features"],
            paraphrase_outputs["delta_features"] if paraphrase_outputs is not None else outputs["delta_features"].new_zeros((0, outputs["delta_features"].size(-1))),
            paraphrase_owner_indices,
            negative_outputs["delta_features"],
            negative_owner_indices,
        )
    else:
        contrastive_loss = text_loss.new_zeros(())

    total_loss = (
        loss_config.text * text_loss
        + loss_config.state * state_loss
        + loss_config.paraphrase * paraphrase_loss
        + loss_config.contrastive * contrastive_loss
    )
    metrics = _compute_batch_metrics(
        model_type,
        outputs,
        batch=batch,
        state_tensorizer=state_tensorizer,
        paraphrase_outputs=paraphrase_outputs,
    )
    return {
        "loss": total_loss,
        "text_loss": text_loss,
        "state_loss": state_loss,
        "paraphrase_loss": paraphrase_loss,
        "contrastive_loss": contrastive_loss,
        **metrics,
    }


def _aggregate_metric_rows(rows: list[dict[str, float | dict[str, float]]]) -> dict[str, Any]:
    if not rows:
        return {}
    total_examples = sum(float(row.get("example_count", 0.0)) for row in rows)
    if total_examples <= 0.0:
        total_examples = float(len(rows))

    weighted: dict[str, float] = {}
    nested_weighted: dict[str, dict[str, float]] = {}
    for row in rows:
        weight = float(row.get("example_count", 0.0)) or 1.0
        for key, value in row.items():
            if isinstance(value, dict):
                target = nested_weighted.setdefault(key, {})
                for nested_key, nested_value in value.items():
                    target[nested_key] = target.get(nested_key, 0.0) + float(nested_value) * weight
            elif isinstance(value, Tensor):
                weighted[key] = weighted.get(key, 0.0) + float(value.detach().cpu().item()) * weight
            elif isinstance(value, (int, float)):
                weighted[key] = weighted.get(key, 0.0) + float(value) * weight
    output: dict[str, Any] = {
        key: value / total_examples
        for key, value in weighted.items()
    }
    for key, nested_values in nested_weighted.items():
        output[key] = {
            nested_key: nested_value / total_examples
            for nested_key, nested_value in nested_values.items()
        }
    return output


def _evaluate(
    model_type: ModelType,
    model: torch.nn.Module,
    dataloader: DataLoader,
    *,
    pad_id: int,
    loss_config: LossConfig,
    device: torch.device,
    state_tensorizer: StateTensorizer,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    model.eval()
    rows: list[dict[str, float | dict[str, float]]] = []
    use_amp = amp_dtype is not None
    with torch.no_grad():
        for batch in dataloader:
            moved_batch = _move_batch_to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                rows.append(
                    _compute_losses(
                        model_type,
                        model,
                        moved_batch,
                        pad_id=pad_id,
                        loss_config=loss_config,
                        state_tensorizer=state_tensorizer,
                    )
                )
    if not rows:
        return {
            "loss": 0.0,
            "text_loss": 0.0,
            "state_loss": 0.0,
            "paraphrase_loss": 0.0,
            "contrastive_loss": 0.0,
            "example_count": 0.0,
            "exact_state_accuracy": 0.0,
            "per_field_accuracy": {
                "holder_accuracy": 0.0,
                "visibility_accuracy": 0.0,
                "container_accuracy": 0.0,
                "macro_accuracy": 0.0,
            },
            "paraphrase_delta_cosine": 0.0,
            "delta_norm_mean": 0.0,
            "delta_norm_std": 0.0,
        }
    return _aggregate_metric_rows(rows)


def _write_json(path: Path, payload: Any) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(stable_json_dumps(payload))
        handle.write("\n")


def _save_artifacts(
    output_dir: Path,
    *,
    config: TrainConfig,
    tokenizer: SimpleTokenizer,
    state_tensorizer: StateTensorizer,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config_snapshot.yaml").open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(config.model_dump(mode="json"), handle, sort_keys=False)
    _write_json(output_dir / "tokenizer.json", tokenizer.state_dict())
    _write_json(output_dir / "state_tensorizer.json", state_tensorizer.state_dict())


def _save_checkpoint(
    path: Path,
    *,
    model_type: ModelType,
    model: torch.nn.Module,
    optimizer: AdamW,
    config: TrainConfig,
    tokenizer: SimpleTokenizer,
    state_tensorizer: StateTensorizer,
    epoch: int,
    global_step: int,
    best_metric_name: str,
    best_metric_value: float | None,
) -> None:
    ensure_parent_dir(path)
    torch.save(
        {
            "model_type": model_type,
            "config": config.model_dump(mode="json"),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "tokenizer": tokenizer.state_dict(),
            "state_tensorizer": state_tensorizer.state_dict(),
            "metadata": {
                "epoch": epoch,
                "global_step": global_step,
                "best_metric_name": best_metric_name,
                "best_metric_value": best_metric_value,
            },
        },
        path,
    )


def _load_checkpoint(
    checkpoint_path: Path,
) -> tuple[dict[str, object], TrainConfig, SimpleTokenizer, StateTensorizer]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = TrainConfig.model_validate(checkpoint["config"])
    tokenizer = SimpleTokenizer.from_state(checkpoint["tokenizer"])
    state_tensorizer = StateTensorizer.from_state(checkpoint["state_tensorizer"])
    return checkpoint, config, tokenizer, state_tensorizer


def _load_checkpoint_for_device(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[dict[str, object], TrainConfig, SimpleTokenizer, StateTensorizer]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = TrainConfig.model_validate(checkpoint["config"])
    tokenizer = SimpleTokenizer.from_state(checkpoint["tokenizer"])
    state_tensorizer = StateTensorizer.from_state(checkpoint["state_tensorizer"])
    return checkpoint, config, tokenizer, state_tensorizer


def _append_train_log(path: Path, row: dict[str, float | int]) -> None:
    ensure_parent_dir(path)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(stable_json_dumps(row))
        handle.write("\n")


def train_model(
    config_path: Path,
    *,
    model_type: ModelType,
) -> dict[str, Any]:
    """Run a training loop for one WDLM benchmark model."""

    config = load_train_config(config_path)
    if config.model_type is not None and config.model_type != model_type:
        raise ValueError(f"Config model_type={config.model_type!r} does not match requested {model_type!r}.")
    set_global_seed(config.seed)
    device = resolve_device(config.device)
    configure_runtime_for_speed(device)
    amp_dtype = resolve_precision(config.precision, device)
    use_amp = amp_dtype is not None
    scaler = GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16)
    print(f"[wdlm] train device: {device}")
    print(f"[wdlm] train precision: {amp_dtype if amp_dtype is not None else 'fp32'}")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_tokenizer: SimpleTokenizer | None = None
    resume_state_tensorizer: StateTensorizer | None = None
    start_epoch = 0
    global_step = 0
    best_metric_value: float | None = None
    checkpoint_model_type = model_type
    optimizer_state: dict[str, Any] | None = None

    if config.optim.resume_from:
        checkpoint, _, resume_tokenizer, resume_state_tensorizer = _load_checkpoint_for_device(
            Path(config.optim.resume_from),
            device=device,
        )
        checkpoint_model_type = checkpoint["model_type"]  # type: ignore[assignment]
        if checkpoint_model_type != model_type:
            raise ValueError("Resume checkpoint model_type does not match requested model_type.")
        metadata = checkpoint["metadata"]  # type: ignore[assignment]
        start_epoch = int(metadata["epoch"]) + 1
        global_step = int(metadata["global_step"])
        best_metric_value = (
            None if metadata["best_metric_value"] is None else float(metadata["best_metric_value"])
        )
        optimizer_state = checkpoint["optimizer_state"]  # type: ignore[assignment]

    train_dataset = _build_dataset(
        config,
        split="train",
        tokenizer=resume_tokenizer,
        state_tensorizer=resume_state_tensorizer,
    )
    val_dataset = _build_dataset(
        config,
        split="val",
        tokenizer=train_dataset.tokenizer,
        state_tensorizer=train_dataset.state_tensorizer,
    )
    _save_artifacts(
        output_dir,
        config=config,
        tokenizer=train_dataset.tokenizer,
        state_tensorizer=train_dataset.state_tensorizer,
    )

    print(
        "[wdlm] dataloader settings: "
        f"batch_size={config.optim.batch_size} num_workers="
        f"{0 if config.data.num_workers is None else config.data.num_workers} "
        f"pin_memory={bool(config.data.pin_memory) if config.data.pin_memory is not None else device.type == 'cuda'}"
    )
    train_loader = _make_dataloader(train_dataset, config, shuffle=True, device=device)
    val_loader = _make_dataloader(val_dataset, config, shuffle=False, device=device)
    model = _build_model(
        model_type,
        vocab_size=train_dataset.tokenizer.vocab_size,
        state_input_dim=train_dataset.state_tensorizer.vector_dim,
        model_config=config.model,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=config.optim.lr)

    if config.optim.resume_from:
        checkpoint, _, _, _ = _load_checkpoint_for_device(
            Path(config.optim.resume_from),
            device=device,
        )
        model.load_state_dict(checkpoint["model_state"])
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)

    best_metric_mode = _resolve_best_metric_mode(config.best_metric, config.best_metric_mode)
    train_log_path = output_dir / "train_log.jsonl"
    last_val_metrics: dict[str, Any] = {}
    last_completed_epoch = max(start_epoch - 1, 0)
    previous_macro_accuracy: float | None = None
    for epoch in range(start_epoch, config.optim.epochs):
        model.train()
        for batch in train_loader:
            moved_batch = _move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss_map = _compute_losses(
                    model_type,
                    model,
                    moved_batch,
                    pad_id=train_dataset.tokenizer.pad_id,
                    loss_config=config.loss,
                    state_tensorizer=train_dataset.state_tensorizer,
                )
            loss_tensor = loss_map["loss"]
            if not isinstance(loss_tensor, Tensor):
                raise TypeError("Expected tensor loss.")
            if scaler.is_enabled():
                scaler.scale(loss_tensor).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_tensor.backward()
                optimizer.step()
            global_step += 1
            train_row = flatten_metric_dict(
                {
                    key: value
                    for key, value in loss_map.items()
                    if key != "loss" or isinstance(value, Tensor)
                }
            )
            train_row["loss"] = float(loss_tensor.detach().cpu().item())
            train_row["epoch"] = epoch
            train_row["step"] = global_step
            _append_train_log(train_log_path, train_row)
            if config.optim.max_steps is not None and global_step >= config.optim.max_steps:
                break
        last_val_metrics = _evaluate(
            model_type,
            model,
            val_loader,
            pad_id=train_dataset.tokenizer.pad_id,
            loss_config=config.loss,
            device=device,
            state_tensorizer=train_dataset.state_tensorizer,
            amp_dtype=amp_dtype,
        )
        last_completed_epoch = epoch
        field_metrics = last_val_metrics.get("per_field_accuracy", {})
        print(
            "[wdlm] val "
            + stable_json_dumps(
                {
                    "epoch": epoch,
                    "step": global_step,
                    "text_loss": last_val_metrics.get("text_loss"),
                    "exact_state_accuracy": last_val_metrics.get("exact_state_accuracy"),
                    "per_field_macro_accuracy": field_metrics.get("macro_accuracy"),
                    "holder_accuracy": field_metrics.get("holder_accuracy"),
                    "visibility_accuracy": field_metrics.get("visibility_accuracy"),
                    "container_accuracy": field_metrics.get("container_accuracy"),
                    "paraphrase_delta_cosine": last_val_metrics.get("paraphrase_delta_cosine"),
                }
            )
        )
        partial_note = _partial_learning_note(last_val_metrics, previous_macro_accuracy)
        if partial_note is not None:
            print(partial_note)
        previous_macro_accuracy = _macro_per_field_accuracy(last_val_metrics)
        current_metric = _extract_metric_value(last_val_metrics, config.best_metric)
        if _metric_better(current_metric, best_metric_value, best_metric_mode):
            best_metric_value = current_metric
            _save_checkpoint(
                output_dir / "checkpoint_best.pt",
                model_type=model_type,
                model=model,
                optimizer=optimizer,
                config=config,
                tokenizer=train_dataset.tokenizer,
                state_tensorizer=train_dataset.state_tensorizer,
                epoch=epoch,
                global_step=global_step,
                best_metric_name=config.best_metric,
                best_metric_value=best_metric_value,
            )
        if config.optim.max_steps is not None and global_step >= config.optim.max_steps:
            break

    final_metrics = _evaluate(
        model_type,
        model,
        val_loader,
        pad_id=train_dataset.tokenizer.pad_id,
        loss_config=config.loss,
        device=device,
        state_tensorizer=train_dataset.state_tensorizer,
        amp_dtype=amp_dtype,
    )
    final_metrics["global_steps"] = float(global_step)
    final_metrics["best_metric_name"] = config.best_metric
    final_metrics["best_metric_value"] = float(
        best_metric_value if best_metric_value is not None else _extract_metric_value(final_metrics, config.best_metric)
    )
    if last_val_metrics:
        final_metrics["last_val"] = last_val_metrics
    _save_checkpoint(
        output_dir / "checkpoint_last.pt",
        model_type=model_type,
        model=model,
        optimizer=optimizer,
        config=config,
        tokenizer=train_dataset.tokenizer,
        state_tensorizer=train_dataset.state_tensorizer,
        epoch=last_completed_epoch,
        global_step=global_step,
        best_metric_name=config.best_metric,
        best_metric_value=best_metric_value,
    )
    _write_json(output_dir / "metrics.json", final_metrics)
    return final_metrics


def eval_model(
    config_path: Path,
    *,
    checkpoint_path: Path,
    split: str = "val",
) -> dict[str, Any]:
    """Evaluate a saved model checkpoint on any configured split."""

    config = load_train_config(config_path)
    device = resolve_device(config.device)
    configure_runtime_for_speed(device)
    amp_dtype = resolve_precision(config.precision, device)
    print(f"[wdlm] eval device: {device}")
    print(f"[wdlm] eval precision: {amp_dtype if amp_dtype is not None else 'fp32'}")
    checkpoint, checkpoint_config, tokenizer, state_tensorizer = _load_checkpoint_for_device(
        checkpoint_path,
        device=device,
    )
    if config.model_type is None:
        config = checkpoint_config
    model_type = checkpoint["model_type"]  # type: ignore[assignment]
    dataset = _build_dataset(
        config,
        split=split,
        tokenizer=tokenizer,
        state_tensorizer=state_tensorizer,
    )
    dataloader = _make_dataloader(dataset, config, shuffle=False, device=device)
    model = _build_model(
        model_type,
        vocab_size=tokenizer.vocab_size,
        state_input_dim=state_tensorizer.vector_dim,
        model_config=config.model,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    metrics = _evaluate(
        model_type,
        model,
        dataloader,
        pad_id=tokenizer.pad_id,
        loss_config=config.loss,
        device=device,
        state_tensorizer=state_tensorizer,
        amp_dtype=amp_dtype,
    )
    metrics["device"] = str(device)
    metrics["split"] = split
    partial_note = _partial_learning_note(metrics)
    if partial_note is not None:
        print(partial_note)
    return metrics


def evaluate_benchmark(
    config_path: Path,
    *,
    checkpoint_path: Path,
    output_dir: Path | None = None,
    splits: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate a checkpoint on one or more benchmark splits and write reports."""

    config = load_train_config(config_path)
    benchmark_splits = splits or config.data.benchmark_splits
    run_dir = output_dir or (Path(config.output_dir) / "benchmark_eval")
    run_dir.mkdir(parents=True, exist_ok=True)

    per_split_metrics = {
        split_name: eval_model(config_path, checkpoint_path=checkpoint_path, split=split_name)
        for split_name in benchmark_splits
    }
    aggregate_rows = []
    for split_name, metrics in per_split_metrics.items():
        row = dict(metrics)
        row["split"] = split_name
        aggregate_rows.append(
            {
                key: value
                for key, value in row.items()
                if key != "split"
            }
        )
    aggregate_metrics = _aggregate_metric_rows(aggregate_rows)
    _write_json(
        run_dir / "per_split_metrics.json",
        per_split_metrics,
    )
    _write_json(
        run_dir / "metrics.json",
        {
            "checkpoint": str(checkpoint_path),
            "splits": benchmark_splits,
            "aggregate_metrics": aggregate_metrics,
        },
    )
    summary_lines = [
        "# WDLM Benchmark Summary",
        "",
        f"Checkpoint: `{checkpoint_path}`",
        "",
        "| split | text_loss | exact_state_accuracy | per_field_macro_accuracy | holder_accuracy | visibility_accuracy | container_accuracy | paraphrase_delta_cosine | example_count |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split_name in benchmark_splits:
        metrics = per_split_metrics[split_name]
        per_field = metrics.get("per_field_accuracy", {})
        summary_lines.append(
            "| "
            + " | ".join(
                [
                    split_name,
                    f"{float(metrics['text_loss']):.4f}",
                    f"{float(metrics['exact_state_accuracy']):.4f}",
                    f"{float(per_field.get('macro_accuracy', 0.0)):.4f}",
                    f"{float(per_field.get('holder_accuracy', 0.0)):.4f}",
                    f"{float(per_field.get('visibility_accuracy', 0.0)):.4f}",
                    f"{float(per_field.get('container_accuracy', 0.0)):.4f}",
                    f"{float(metrics['paraphrase_delta_cosine']):.4f}",
                    f"{float(metrics['example_count']):.0f}",
                ]
            )
            + " |"
        )
    with (run_dir / "summary.md").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(summary_lines))
        handle.write("\n")
    return {
        "aggregate_metrics": aggregate_metrics,
        "per_split_metrics": per_split_metrics,
        "summary_path": str(run_dir / "summary.md"),
    }


def run_experiment(config_path: Path) -> dict[str, Any]:
    """Train from config, then evaluate the best checkpoint on benchmark splits."""

    config = load_train_config(config_path)
    if config.model_type is None:
        raise ValueError("Config must define model_type for run_experiment.")
    train_metrics = train_model(config_path, model_type=config.model_type)
    benchmark_metrics = evaluate_benchmark(
        config_path,
        checkpoint_path=Path(config.output_dir) / "checkpoint_best.pt",
        output_dir=Path(config.output_dir) / "benchmark_eval",
    )
    report = {
        "train_metrics": train_metrics,
        "benchmark": benchmark_metrics,
    }
    _write_json(Path(config.output_dir) / "experiment_report.json", report)
    return report
