from __future__ import annotations

from pathlib import Path

from wdlm.data.generate import generate_toy_world_file
from wdlm.models.para_align_baseline import ParaphraseAlignedBaselineModel
from wdlm.models.state_head_baseline import StateHeadBaselineModel
from wdlm.train.collate import collate_examples
from wdlm.train.dataset import StepDataset


def test_state_head_baseline_forward_shapes(tmp_path: Path) -> None:
    data_path = tmp_path / "steps.jsonl"
    generate_toy_world_file(data_path, num_examples=6, seed=13)
    dataset = StepDataset(data_path, max_vocab_size=128, max_seq_len=32)
    batch = collate_examples(
        [dataset[index] for index in range(3)],
        pad_id=dataset.tokenizer.pad_id,
        max_paraphrases=2,
    )

    model = StateHeadBaselineModel(
        vocab_size=dataset.tokenizer.vocab_size,
        state_input_dim=dataset.state_tensorizer.vector_dim,
        d_model=32,
        n_heads=4,
        n_layers=2,
        ffn_dim=64,
        max_seq_len=32,
        dropout=0.1,
        state_dim=24,
    )
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        state_before=batch["state_before"],
    )

    assert outputs["logits"].shape[:2] == batch["input_ids"].shape
    assert outputs["state_logits"].shape == batch["state_after"].shape
    assert outputs["delta_features"].shape == (3, 24)


def test_para_align_baseline_forward_shapes(tmp_path: Path) -> None:
    data_path = tmp_path / "steps.jsonl"
    generate_toy_world_file(data_path, num_examples=6, seed=17)
    dataset = StepDataset(data_path, max_vocab_size=128, max_seq_len=32)
    batch = collate_examples(
        [dataset[index] for index in range(3)],
        pad_id=dataset.tokenizer.pad_id,
        max_paraphrases=2,
    )

    model = ParaphraseAlignedBaselineModel(
        vocab_size=dataset.tokenizer.vocab_size,
        state_input_dim=dataset.state_tensorizer.vector_dim,
        d_model=32,
        n_heads=4,
        n_layers=2,
        ffn_dim=64,
        max_seq_len=32,
        dropout=0.1,
        state_dim=24,
    )
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        state_before=batch["state_before"],
    )

    assert outputs["logits"].shape[:2] == batch["input_ids"].shape
    assert outputs["state_logits"].shape == batch["state_after"].shape
    assert outputs["delta_features"].shape == (3, 24)
