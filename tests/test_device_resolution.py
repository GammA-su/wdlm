from __future__ import annotations

import pytest
import torch

from wdlm.train.trainer import _move_batch_to_device, resolve_device


def test_resolve_device_auto_uses_cpu_when_cuda_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto").type == "cpu"


def test_resolve_device_explicit_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("cpu").type == "cpu"


def test_resolve_device_explicit_cuda_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        resolve_device("cuda")


def test_move_batch_to_device_moves_tensor_values() -> None:
    batch = {
        "input_ids": torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
        "labels": torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
        "example_ids": ["a", "b"],
    }
    moved = _move_batch_to_device(batch, torch.device("cpu"))
    assert isinstance(moved["input_ids"], torch.Tensor)
    assert isinstance(moved["labels"], torch.Tensor)
    assert moved["input_ids"].device.type == "cpu"
    assert moved["labels"].device.type == "cpu"
    assert moved["example_ids"] == ["a", "b"]
