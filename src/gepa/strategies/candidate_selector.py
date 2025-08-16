# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import random

from gepa.core.state import GEPAState
from gepa.gepa_utils import (
    idxmax,
    select_program_candidate_from_pareto_front,
    select_program_candidate_from_objective_pareto_front,
    select_program_candidate_from_hybrid_front,
)
from gepa.proposer.reflective_mutation.base import CandidateSelector


class ParetoCandidateSelector(CandidateSelector):
    def __init__(
        self,
        rng: random.Random | None,
        selection_strategy: str = "auto",
        objectives: list[str] | None = None,
        normalize: str = "zscore",
        global_bonus: int = 3,
        instance_sample_size: int | None = None,
    ):
        self.rng = rng or random.Random(0)
        self.selection_strategy = selection_strategy
        self.objectives = objectives
        self.normalize = normalize
        self.global_bonus = global_bonus
        self.instance_sample_size = instance_sample_size

    def select_candidate_idx(self, state: GEPAState) -> int:
        strategy = self.selection_strategy
        if strategy == "auto":
            strategy = "objective_pareto" if state.is_multi_objective else "instance_pareto"

        assert len(state.per_program_tracked_scores) == len(state.program_candidates)

        if strategy == "instance_pareto":
            return select_program_candidate_from_pareto_front(
                state.program_at_pareto_front_valset,
                state.per_program_tracked_scores,
                self.rng,
            )
        elif strategy == "objective_pareto":
            assert state.program_objective_scores_val_set is not None
            objs = self.objectives or state.objective_names or []
            return select_program_candidate_from_objective_pareto_front(
                state.program_objective_scores_val_set,
                self.rng,
                self.normalize,
                objs,
            )
        elif strategy == "hybrid":
            assert state.program_objective_scores_val_set is not None
            objs = self.objectives or state.objective_names or []
            return select_program_candidate_from_hybrid_front(
                state.program_at_pareto_front_valset,
                state.program_objective_scores_val_set,
                self.rng,
                self.normalize,
                objs,
                self.global_bonus,
                self.instance_sample_size,
            )
        else:
            raise ValueError(f"Unknown selection strategy: {strategy}")

class CurrentBestCandidateSelector(CandidateSelector):
    def __init__(self):
        pass

    def select_candidate_idx(self, state: GEPAState) -> int:
        assert len(state.per_program_tracked_scores) == len(state.program_candidates)
        return idxmax(state.per_program_tracked_scores)
