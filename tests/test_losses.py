from __future__ import annotations

import math
from pathlib import Path

import torch

from wdlm.data.generate import generate_toy_world_file
from wdlm.data.split import split_dataset_file
from wdlm.models.baseline_lm import BaselineLanguageModel
from wdlm.models.wdlm import WDLMModel
from wdlm.train.collate import collate_examples
from wdlm.train.dataset import StepDataset
from wdlm.train.losses import (
    causal_text_cross_entropy,
    paraphrase_delta_invariance_loss,
    state_mse_loss,
    supervised_contrastive_delta_loss,
)


def test_losses_are_finite_and_manual_optimizer_steps_succeed(tmp_path: Path) -> None:
    data_path = tmp_path / "steps.jsonl"
    splits_dir = tmp_path / "splits"
    generate_toy_world_file(data_path, num_examples=8, seed=9)
    split_paths = split_dataset_file(input_path=data_path, out_dir=splits_dir, seed=9)

    dataset = StepDataset(split_paths["train"], max_vocab_size=64, max_seq_len=24)
    batch = collate_examples(
        [dataset[index] for index in range(min(2, len(dataset)))],
        pad_id=dataset.tokenizer.pad_id,
        max_paraphrases=1,
    )
    model = WDLMModel(
        vocab_size=dataset.tokenizer.vocab_size,
        state_input_dim=dataset.state_tensorizer.vector_dim,
        d_model=16,
        n_heads=2,
        n_layers=1,
        ffn_dim=32,
        max_seq_len=24,
        dropout=0.1,
        state_dim=12,
    )
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        state_before=batch["state_before"],
    )
    with torch.no_grad():
        target_state = model.encode_state(batch["state_after"])
    losses = [
        causal_text_cross_entropy(outputs["logits"], batch["labels"], pad_id=dataset.tokenizer.pad_id),
        state_mse_loss(outputs["s_pred"], target_state),
        paraphrase_delta_invariance_loss(
            outputs["delta_features"],
            outputs["delta_features"].index_select(0, torch.zeros(outputs["delta_features"].size(0), dtype=torch.long)),
            torch.zeros(outputs["delta_features"].size(0), dtype=torch.long),
        ),
        supervised_contrastive_delta_loss(
            outputs["delta_features"],
            outputs["delta_features"].index_select(0, torch.zeros(outputs["delta_features"].size(0), dtype=torch.long)),
            torch.zeros(outputs["delta_features"].size(0), dtype=torch.long),
            outputs["delta_features"].index_select(0, torch.zeros(outputs["delta_features"].size(0), dtype=torch.long)),
            torch.zeros(outputs["delta_features"].size(0), dtype=torch.long),
        ),
    ]
    assert all(math.isfinite(float(loss.item())) for loss in losses)

    baseline = BaselineLanguageModel(
        vocab_size=dataset.tokenizer.vocab_size,
        d_model=16,
        n_heads=2,
        n_layers=1,
        ffn_dim=32,
        max_seq_len=24,
        dropout=0.1,
    )
    baseline_optimizer = torch.optim.AdamW(baseline.parameters(), lr=1e-3)
    baseline_optimizer.zero_grad(set_to_none=True)
    baseline_outputs = baseline(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    baseline_loss = causal_text_cross_entropy(
        baseline_outputs["logits"],
        batch["labels"],
        pad_id=dataset.tokenizer.pad_id,
    )
    baseline_loss.backward()
    baseline_optimizer.step()
    assert math.isfinite(float(baseline_loss.item()))

    wdlm_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    wdlm_optimizer.zero_grad(set_to_none=True)
    train_outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        state_before=batch["state_before"],
    )
    train_loss = (
        causal_text_cross_entropy(train_outputs["logits"], batch["labels"], pad_id=dataset.tokenizer.pad_id)
        + state_mse_loss(train_outputs["s_pred"], model.encode_state(batch["state_after"]).detach())
    )
    train_loss.backward()
    wdlm_optimizer.step()
    assert math.isfinite(float(train_loss.item()))
