"""Causal LM baseline with a paraphrase-aligned representation head."""

from __future__ import annotations

from wdlm.models.state_head_baseline import StateHeadBaselineModel


class ParaphraseAlignedBaselineModel(StateHeadBaselineModel):
    """State-head baseline used with paraphrase and contrastive alignment losses."""

