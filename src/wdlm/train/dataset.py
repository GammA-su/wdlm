"""Dataset loading, tokenization, and state tensorization."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset

from wdlm.schemas import ExampleRecord, WorldState
from wdlm.utils.io import read_jsonl


ACTION_TYPES: tuple[str, ...] = ("close", "give", "hide", "move", "open", "reveal")
ACTION_TYPE_TO_ID = {name: index for index, name in enumerate(ACTION_TYPES)}
VISIBILITY_TO_INDEX = {"visible": 0, "hidden": 1}
CONTAINER_STATUS_TO_INDEX = {"open": 0, "closed": 1}
SPECIAL_TOKENS: tuple[str, ...] = ("<pad>", "<bos>", "<eos>", "<unk>")


def tokenize_text(text: str) -> list[str]:
    """Tokenize text with a small regex-based tokenizer."""

    return re.findall(r"[A-Za-z_]+|[0-9]+|[^\w\s]", text.lower())


@dataclass
class SimpleTokenizer:
    """A tiny regex tokenizer with a dataset-built vocabulary."""

    token_to_id: dict[str, int]
    max_seq_len: int

    @classmethod
    def build_from_examples(
        cls,
        examples: list[ExampleRecord],
        *,
        max_vocab_size: int,
        max_seq_len: int,
    ) -> "SimpleTokenizer":
        counts: Counter[str] = Counter()
        for example in examples:
            counts.update(tokenize_text(example.text_chunk))
            for paraphrase in example.paraphrases:
                counts.update(tokenize_text(paraphrase))
            for negative_update in example.negative_updates:
                counts.update(tokenize_text(negative_update.text_chunk))
        vocab_tokens = list(SPECIAL_TOKENS)
        for token, _ in counts.most_common(max_vocab_size - len(SPECIAL_TOKENS)):
            if token not in vocab_tokens:
                vocab_tokens.append(token)
        return cls(
            token_to_id={token: index for index, token in enumerate(vocab_tokens)},
            max_seq_len=max_seq_len,
        )

    @classmethod
    def from_state(cls, state: dict[str, object]) -> "SimpleTokenizer":
        return cls(
            token_to_id={str(key): int(value) for key, value in state["token_to_id"].items()},
            max_seq_len=int(state["max_seq_len"]),
        )

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<unk>"]

    def encode(self, text: str) -> list[int]:
        ids = [self.bos_id]
        ids.extend(self.token_to_id.get(token, self.unk_id) for token in tokenize_text(text))
        ids.append(self.eos_id)
        if len(ids) > self.max_seq_len:
            ids = ids[: self.max_seq_len]
            ids[-1] = self.eos_id
        return ids

    def state_dict(self) -> dict[str, object]:
        return {"token_to_id": self.token_to_id, "max_seq_len": self.max_seq_len}


@dataclass
class StateTensorizer:
    """Encode explicit toy-world states into compact float vectors."""

    objects: list[str]
    holders: list[str]
    containers: list[str]

    @classmethod
    def build_from_examples(cls, examples: list[ExampleRecord]) -> "StateTensorizer":
        object_names = sorted(
            {
                object_name
                for example in examples
                for object_name in example.state_before.objects
            }
        )
        holders = sorted(
            {
                holder
                for example in examples
                for state in (example.state_before, example.state_after)
                for holder in (
                    list(state.locations)
                    + list(state.owners)
                    + list(state.containers.keys())
                )
            }
        )
        containers = sorted(
            {
                container
                for example in examples
                for state in (example.state_before, example.state_after)
                for container in state.containers
            }
        )
        return cls(objects=object_names, holders=holders, containers=containers)

    @classmethod
    def from_state(cls, state: dict[str, object]) -> "StateTensorizer":
        return cls(
            objects=list(state["objects"]),
            holders=list(state["holders"]),
            containers=list(state["containers"]),
        )

    @property
    def vector_dim(self) -> int:
        return len(self.objects) * len(self.holders) + len(self.objects) * 2 + len(self.containers) * 2

    def iter_field_slices(self) -> list[tuple[str, Literal["holder", "visibility", "container"], slice]]:
        """Return deterministic field slices for decoding and metrics."""

        fields: list[tuple[str, Literal["holder", "visibility", "container"], slice]] = []
        holder_width = len(self.holders)
        holder_offset = 0
        visibility_offset = len(self.objects) * holder_width
        container_offset = visibility_offset + len(self.objects) * 2
        for object_index, object_name in enumerate(self.objects):
            holder_start = holder_offset + object_index * holder_width
            fields.append(
                (
                    f"holder:{object_name}",
                    "holder",
                    slice(holder_start, holder_start + holder_width),
                )
            )
            visibility_start = visibility_offset + object_index * 2
            fields.append(
                (
                    f"visibility:{object_name}",
                    "visibility",
                    slice(visibility_start, visibility_start + 2),
                )
            )
        for container_index, container_name in enumerate(self.containers):
            container_start = container_offset + container_index * 2
            fields.append(
                (
                    f"container:{container_name}",
                    "container",
                    slice(container_start, container_start + 2),
                )
            )
        return fields

    def decode_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """Project raw scores into a one-hot explicit state prediction tensor."""

        if scores.ndim == 1:
            scores = scores.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        decoded = torch.zeros_like(scores)
        for _, _, field_slice in self.iter_field_slices():
            field_scores = scores[:, field_slice]
            best_indices = field_scores.argmax(dim=-1, keepdim=True)
            decoded[:, field_slice].scatter_(1, best_indices, 1.0)
        return decoded.squeeze(0) if squeeze else decoded

    def encode_state(self, state: WorldState) -> torch.Tensor:
        holder_offset = 0
        visibility_offset = len(self.objects) * len(self.holders)
        container_offset = visibility_offset + len(self.objects) * 2
        holder_to_index = {holder: index for index, holder in enumerate(self.holders)}
        container_to_index = {container: index for index, container in enumerate(self.containers)}
        vector = torch.zeros(self.vector_dim, dtype=torch.float32)
        for object_index, object_name in enumerate(self.objects):
            object_state = state.objects.get(object_name)
            if object_state is None:
                continue
            holder_index = holder_to_index[object_state.holder]
            vector[holder_offset + object_index * len(self.holders) + holder_index] = 1.0
            visibility_index = VISIBILITY_TO_INDEX[object_state.visibility]
            vector[visibility_offset + object_index * 2 + visibility_index] = 1.0
        for container_name in self.containers:
            if container_name not in state.containers:
                continue
            status = state.containers[container_name]
            container_index = container_to_index[container_name]
            status_index = CONTAINER_STATUS_TO_INDEX[status]
            vector[container_offset + container_index * 2 + status_index] = 1.0
        return vector

    def state_dict(self) -> dict[str, object]:
        return {
            "objects": self.objects,
            "holders": self.holders,
            "containers": self.containers,
        }


class StepDataset(Dataset[dict[str, object]]):
    """Load step-level JSONL examples for baseline and WDLM training."""

    def __init__(
        self,
        path: Path,
        *,
        tokenizer: SimpleTokenizer | None = None,
        state_tensorizer: StateTensorizer | None = None,
        max_vocab_size: int = 512,
        max_seq_len: int = 64,
    ) -> None:
        self.path = path
        self.examples = [ExampleRecord.model_validate(row) for row in read_jsonl(path)]
        self.tokenizer = tokenizer or SimpleTokenizer.build_from_examples(
            self.examples,
            max_vocab_size=max_vocab_size,
            max_seq_len=max_seq_len,
        )
        self.state_tensorizer = state_tensorizer or StateTensorizer.build_from_examples(self.examples)
        self.encoded_examples = [self._encode_example(example) for example in self.examples]

    def _encode_example(self, example: ExampleRecord) -> dict[str, object]:
        return {
            "example_id": example.example_id,
            "input_ids": self.tokenizer.encode(example.text_chunk),
            "paraphrase_ids": [self.tokenizer.encode(text) for text in example.paraphrases],
            "negative_ids": [
                self.tokenizer.encode(negative.text_chunk)
                for negative in example.negative_updates
            ],
            "state_before": self.state_tensorizer.encode_state(example.state_before),
            "state_after": self.state_tensorizer.encode_state(example.state_after),
            "action_type_id": ACTION_TYPE_TO_ID[example.action_struct.type],
        }

    def __len__(self) -> int:
        return len(self.encoded_examples)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.encoded_examples[index]
