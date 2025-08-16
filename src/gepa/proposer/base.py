# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from dataclasses import dataclass, field
from typing import Any, Protocol

from gepa.core.state import GEPAState
from gepa.gepa_utils import Score


@dataclass
class CandidateProposal:
    candidate: dict[str, str]
    parent_program_ids: list[int]
    # Optional mini-batch / subsample info
    subsample_indices: list[int] | None = None
    subsample_scores_before: list[Score] | None = None
    subsample_scores_after: list[Score] | None = None
    # Free-form metadata for logging/trace
    tag: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class ProposeNewCandidate(Protocol):
    """
    Strategy that receives the current optimizer state and proposes a new candidate or returns None.
    It may compute subsample evaluations, set trace fields in state, etc.
    The engine will handle acceptance and full eval unless the strategy already did those and encoded in metadata.
    """
    def propose(self, state: GEPAState) -> CandidateProposal | None:
        ...
