"""Typed schemas for WDLM synthetic data."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Mapping, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)


DifficultyLevel: TypeAlias = Literal["easy", "medium", "hard"]
Visibility: TypeAlias = Literal["visible", "hidden"]
ContainerStatus: TypeAlias = Literal["open", "closed"]
NegativeType: TypeAlias = Literal[
    "wrong_object",
    "wrong_destination",
    "opposite_operation",
    "wrong_owner",
    "fallback",
]
QuestionType: TypeAlias = Literal[
    "where_is_object",
    "who_owns_object",
    "is_container_open",
    "is_object_visible",
]


class ObjectState(BaseModel):
    """State for a single object in the toy world."""

    model_config = ConfigDict(extra="forbid")

    holder: str
    visibility: Visibility


class WorldState(BaseModel):
    """Canonical representation of the toy world state."""

    model_config = ConfigDict(extra="forbid")

    locations: list[str]
    owners: list[str]
    containers: dict[str, ContainerStatus]
    objects: dict[str, ObjectState]

    @field_validator("locations", "owners", mode="before")
    @classmethod
    def _sort_names(cls, value: list[str] | tuple[str, ...]) -> list[str]:
        return sorted(dict.fromkeys(value))

    @field_validator("containers", mode="before")
    @classmethod
    def _sort_containers(
        cls,
        value: Mapping[str, ContainerStatus] | dict[str, ContainerStatus],
    ) -> dict[str, ContainerStatus]:
        return dict(sorted(dict(value).items()))

    @field_validator("objects", mode="before")
    @classmethod
    def _sort_objects(
        cls,
        value: Mapping[str, ObjectState | Mapping[str, Any]]
        | dict[str, ObjectState | Mapping[str, Any]],
    ) -> dict[str, ObjectState | Mapping[str, Any]]:
        return dict(sorted(dict(value).items()))


class MoveAction(BaseModel):
    """Move an object between holders."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    type: Literal["move"] = "move"
    object: str
    from_: str = Field(alias="from")
    to: str


class OpenAction(BaseModel):
    """Open a container."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["open"] = "open"
    container: str


class CloseAction(BaseModel):
    """Close a container."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["close"] = "close"
    container: str


class GiveAction(BaseModel):
    """Transfer an object from one owner to another."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["give"] = "give"
    object: str
    from_owner: str
    to_owner: str


class HideAction(BaseModel):
    """Mark an object as hidden."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["hide"] = "hide"
    object: str


class RevealAction(BaseModel):
    """Mark an object as visible."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["reveal"] = "reveal"
    object: str


ActionStruct: TypeAlias = Annotated[
    MoveAction | OpenAction | CloseAction | GiveAction | HideAction | RevealAction,
    Field(discriminator="type"),
]

ACTION_MODEL_TYPES = (
    MoveAction,
    OpenAction,
    CloseAction,
    GiveAction,
    HideAction,
    RevealAction,
)

ACTION_STRUCT_ADAPTER: TypeAdapter[ActionStruct] = TypeAdapter(ActionStruct)


def parse_action_struct(data: ActionStruct | Mapping[str, Any]) -> ActionStruct:
    """Parse an action structure from a dict or already-validated action."""

    if isinstance(data, ACTION_MODEL_TYPES):
        return data
    return ACTION_STRUCT_ADAPTER.validate_python(data)


class NegativeUpdate(BaseModel):
    """A plausible but semantically different action/text pair."""

    model_config = ConfigDict(extra="forbid")

    action_struct: ActionStruct
    text_chunk: str
    negative_type: NegativeType
    template_id: str


class ExampleMetadata(BaseModel):
    """Metadata carried with each dataset step example."""

    model_config = ConfigDict(extra="forbid")

    split: str
    template_id: str
    paraphrase_template_ids: list[str] = Field(default_factory=list)
    difficulty: DifficultyLevel
    seed: int


class ExampleRecord(BaseModel):
    """One per-step JSONL training/evaluation example."""

    model_config = ConfigDict(extra="forbid")

    example_id: str
    world_id: str
    trajectory_id: str
    step_index: int = Field(ge=0)
    episode_length: int = Field(ge=1)
    state_before: WorldState
    action_struct: ActionStruct
    text_chunk: str
    paraphrases: list[str]
    negative_updates: list[NegativeUpdate]
    state_after: WorldState
    metadata: ExampleMetadata

    @model_validator(mode="after")
    def _check_alignment(self) -> "ExampleRecord":
        if len(self.paraphrases) != len(self.metadata.paraphrase_template_ids):
            raise ValueError(
                "Paraphrases and metadata.paraphrase_template_ids must have equal length."
            )
        return self


class TrajectoryMetadata(BaseModel):
    """Metadata for a full trajectory record."""

    model_config = ConfigDict(extra="forbid")

    split: str
    difficulty: DifficultyLevel
    seed: int
    episode_length: int = Field(ge=1)


class TrajectoryRecord(BaseModel):
    """A complete multi-step trajectory."""

    model_config = ConfigDict(extra="forbid")

    trajectory_id: str
    world_id: str
    initial_state: WorldState
    final_state: WorldState
    steps: list[ExampleRecord]
    metadata: TrajectoryMetadata

    @model_validator(mode="after")
    def _check_steps(self) -> "TrajectoryRecord":
        if len(self.steps) != self.metadata.episode_length:
            raise ValueError("Trajectory step count must equal metadata.episode_length.")
        for index, step in enumerate(self.steps):
            if step.trajectory_id != self.trajectory_id:
                raise ValueError("All trajectory steps must share trajectory_id.")
            if step.world_id != self.world_id:
                raise ValueError("All trajectory steps must share world_id.")
            if step.step_index != index:
                raise ValueError("Trajectory steps must be ordered by step_index.")
        return self


class QueryMetadata(BaseModel):
    """Metadata for derived state-query examples."""

    model_config = ConfigDict(extra="forbid")

    split: str
    difficulty: DifficultyLevel
    seed: int


class QueryRecord(BaseModel):
    """A question-answer example derived from a world state."""

    model_config = ConfigDict(extra="forbid")

    qa_id: str
    example_id: str
    world_id: str
    trajectory_id: str
    step_index: int = Field(ge=0)
    question_type: QuestionType
    question: str
    answer: str
    metadata: QueryMetadata


class StatePredictionRecord(BaseModel):
    """Minimal schema for exact-state evaluation inputs."""

    model_config = ConfigDict(extra="forbid")

    example_id: str
    state_after: WorldState
