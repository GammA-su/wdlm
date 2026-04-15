"""Toy object-world state helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from wdlm.schemas import DifficultyLevel, ObjectState, WorldState
from wdlm.utils.rng import SeededRNG


OBJECT_POOL: tuple[str, ...] = (
    "apple",
    "blue_coin",
    "book",
    "red_key",
    "green_block",
    "silver_ring",
    "map",
    "lantern",
)
LOCATION_POOL: tuple[str, ...] = ("bench", "desk", "floor", "shelf", "table")
CONTAINER_POOL: tuple[str, ...] = ("cabinet", "chest", "drawer")
OWNER_POOL: tuple[str, ...] = ("alice", "bob", "carol")

DEFAULT_OBJECTS: tuple[str, ...] = OBJECT_POOL[:4]
DEFAULT_LOCATIONS: tuple[str, ...] = tuple(sorted(("floor", "shelf", "table")))
DEFAULT_CONTAINERS: tuple[str, ...] = tuple(sorted(("cabinet", "drawer")))
DEFAULT_OWNERS: tuple[str, ...] = tuple(sorted(("alice", "bob")))


@dataclass(frozen=True)
class DifficultyConfig:
    """Configuration for one difficulty tier."""

    difficulty: DifficultyLevel
    object_count: int
    location_count: int
    container_count: int
    owner_count: int
    min_episode_length: int
    max_episode_length: int
    allowed_action_types: tuple[str, ...]
    allow_owner_holders: bool
    allow_visibility_actions: bool


DIFFICULTY_CONFIGS: dict[DifficultyLevel, DifficultyConfig] = {
    "easy": DifficultyConfig(
        difficulty="easy",
        object_count=3,
        location_count=2,
        container_count=1,
        owner_count=2,
        min_episode_length=3,
        max_episode_length=4,
        allowed_action_types=("move", "open", "close"),
        allow_owner_holders=False,
        allow_visibility_actions=False,
    ),
    "medium": DifficultyConfig(
        difficulty="medium",
        object_count=4,
        location_count=3,
        container_count=2,
        owner_count=2,
        min_episode_length=5,
        max_episode_length=8,
        allowed_action_types=("move", "open", "close", "give", "hide", "reveal"),
        allow_owner_holders=True,
        allow_visibility_actions=True,
    ),
    "hard": DifficultyConfig(
        difficulty="hard",
        object_count=6,
        location_count=4,
        container_count=3,
        owner_count=3,
        min_episode_length=8,
        max_episode_length=12,
        allowed_action_types=("move", "open", "close", "give", "hide", "reveal"),
        allow_owner_holders=True,
        allow_visibility_actions=True,
    ),
}


@dataclass(frozen=True)
class WorldProfile:
    """Active entities and constraints for one generated world."""

    difficulty: DifficultyLevel
    objects: tuple[str, ...]
    locations: tuple[str, ...]
    containers: tuple[str, ...]
    owners: tuple[str, ...]
    min_episode_length: int
    max_episode_length: int
    allowed_action_types: tuple[str, ...]
    allow_owner_holders: bool
    allow_visibility_actions: bool

    @property
    def holders(self) -> tuple[str, ...]:
        """Return all holders allowed in the current profile."""

        return self.locations + self.containers + self.owners


def get_difficulty_config(difficulty: DifficultyLevel) -> DifficultyConfig:
    """Return the config for a difficulty tier."""

    return DIFFICULTY_CONFIGS[difficulty]


def _sample_names(
    rng: SeededRNG,
    pool: Sequence[str],
    count: int,
    *,
    label: str,
) -> tuple[str, ...]:
    values = rng.derive(label).shuffle(pool)[:count]
    return tuple(sorted(values))


def build_world_profile(rng: SeededRNG, difficulty: DifficultyLevel) -> WorldProfile:
    """Build the active world profile for one generated example or trajectory."""

    config = get_difficulty_config(difficulty)
    return WorldProfile(
        difficulty=difficulty,
        objects=_sample_names(rng, OBJECT_POOL, config.object_count, label="objects"),
        locations=_sample_names(rng, LOCATION_POOL, config.location_count, label="locations"),
        containers=_sample_names(
            rng,
            CONTAINER_POOL,
            config.container_count,
            label="containers",
        ),
        owners=_sample_names(rng, OWNER_POOL, config.owner_count, label="owners"),
        min_episode_length=config.min_episode_length,
        max_episode_length=config.max_episode_length,
        allowed_action_types=config.allowed_action_types,
        allow_owner_holders=config.allow_owner_holders,
        allow_visibility_actions=config.allow_visibility_actions,
    )


def canonicalize_state(state: WorldState | Mapping[str, Any]) -> WorldState:
    """Validate and return a canonically ordered world state."""

    if isinstance(state, WorldState):
        payload = state.model_dump(mode="json")
    else:
        payload = dict(state)
    return WorldState.model_validate(payload)


def is_container(state: WorldState, name: str) -> bool:
    """Return whether a holder name is a container in the current state."""

    current_state = canonicalize_state(state)
    return name in current_state.containers


def is_location(state: WorldState, name: str) -> bool:
    """Return whether a holder name is a location in the current state."""

    current_state = canonicalize_state(state)
    return name in current_state.locations


def is_owner(state: WorldState, name: str) -> bool:
    """Return whether a holder name is an owner/entity in the current state."""

    current_state = canonicalize_state(state)
    return name in current_state.owners


def is_holder(state: WorldState, name: str) -> bool:
    """Return whether a name is any valid holder in the current state."""

    return is_location(state, name) or is_container(state, name) or is_owner(state, name)


def all_holders(state: WorldState) -> tuple[str, ...]:
    """Return every valid holder for objects in the current state."""

    current_state = canonicalize_state(state)
    return tuple(current_state.locations) + tuple(current_state.containers) + tuple(
        current_state.owners
    )


def generate_world_state(
    rng: SeededRNG,
    difficulty: DifficultyLevel = "medium",
    *,
    profile: WorldProfile | None = None,
) -> WorldState:
    """Generate a deterministic random toy-world state."""

    active_profile = profile if profile is not None else build_world_profile(rng, difficulty)
    container_rng = rng.derive("containers")
    containers = {
        container: "open" if container_rng.coin_flip() else "closed"
        for container in active_profile.containers
    }
    holder_choices = list(active_profile.locations) + list(active_profile.containers)
    if active_profile.allow_owner_holders:
        holder_choices.extend(active_profile.owners)

    objects: dict[str, ObjectState] = {}
    object_rng = rng.derive("object-placements")
    for object_name in active_profile.objects:
        holder = object_rng.choice(holder_choices)
        if holder in containers and containers[holder] == "closed":
            visibility = "hidden"
        elif active_profile.allow_visibility_actions:
            visibility = "visible" if object_rng.randint(0, 3) != 0 else "hidden"
        else:
            visibility = "visible"
        objects[object_name] = ObjectState(holder=holder, visibility=visibility)

    return canonicalize_state(
        WorldState(
            locations=list(active_profile.locations),
            owners=list(active_profile.owners),
            containers=containers,
            objects=objects,
        )
    )


def episode_length_for_profile(rng: SeededRNG, profile: WorldProfile) -> int:
    """Return a deterministic episode length for the active profile."""

    return rng.randint(profile.min_episode_length, profile.max_episode_length)


def owner_for_object(state: WorldState, object_name: str) -> str | None:
    """Return the owning entity for an object, if any."""

    current_state = canonicalize_state(state)
    holder = current_state.objects[object_name].holder
    if holder in current_state.owners:
        return holder
    return None
