from __future__ import annotations

from pathlib import Path

from wdlm.data.generate import generate_toy_world_file
from wdlm.models.wdlm import WDLMModel
from wdlm.train.collate import collate_examples
from wdlm.train.dataset import StepDataset


def test_wdlm_forward_shapes(tmp_path: Path) -> None:
    data_path = tmp_path / "steps.jsonl"
    generate_toy_world_file(data_path, num_examples=6, seed=7)
    dataset = StepDataset(data_path, max_vocab_size=128, max_seq_len=32)
    batch = collate_examples(
        [dataset[index] for index in range(3)],
        pad_id=dataset.tokenizer.pad_id,
        max_paraphrases=2,
    )

    model = WDLMModel(
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
    assert outputs["logits"].shape[2] == dataset.tokenizer.vocab_size
    assert outputs["delta_hat"].shape == (3, 24)
    assert outputs["s_pred"].shape == (3, 24)
    assert outputs["state_logits"].shape == batch["state_after"].shape
