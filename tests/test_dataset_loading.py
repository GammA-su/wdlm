from __future__ import annotations

from pathlib import Path

from wdlm.data.generate import generate_toy_world_file
from wdlm.train.collate import collate_examples
from wdlm.train.dataset import StepDataset


def test_step_dataset_and_collate_shapes(tmp_path: Path) -> None:
    data_path = tmp_path / "steps.jsonl"
    generate_toy_world_file(data_path, num_examples=8, seed=42)

    dataset = StepDataset(data_path, max_vocab_size=128, max_seq_len=32)
    samples = [dataset[index] for index in range(2)]
    batch = collate_examples(samples, pad_id=dataset.tokenizer.pad_id, max_paraphrases=2)

    assert len(dataset) == 8
    assert batch["input_ids"].shape[0] == 2
    assert batch["labels"].shape == batch["input_ids"].shape
    assert batch["state_before"].shape == batch["state_after"].shape
    assert batch["state_before"].shape[1] == dataset.state_tensorizer.vector_dim
    assert batch["paraphrase_owner_indices"].ndim == 1
