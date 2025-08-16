# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import random

from gepa.core.state import GEPAState
from gepa.gepa_utils import (
    idxmax,
    select_program_candidate_from_pareto_front,
    Score,
)
from gepa.proposer.reflective_mutation.base import CandidateSelector


class ParetoCandidateSelector(CandidateSelector):
    def __init__(self, rng: random.Random | None):
        if rng is None:
            self.rng = random.Random(0)
        else:
            self.rng = rng

    def select_candidate_idx(self, state: GEPAState) -> int:
        assert len(state.per_program_tracked_scores) == len(state.program_candidates)
        return select_program_candidate_from_pareto_front(
            state.program_at_pareto_front_valset,
            state.per_program_tracked_scores,
            self.rng,
        )


def _normalize_matrix(matrix: list[list[float]], method: str) -> list[list[float]]:
    cols = list(zip(*matrix))
    norm_cols = []
    for col in cols:
        if method == "minmax":
            mn, mx = min(col), max(col)
            if mx == mn:
                norm_cols.append([0.0] * len(col))
            else:
                norm_cols.append([(x - mn) / (mx - mn) for x in col])
        else:  # zscore
            mean = sum(col) / len(col)
            var = sum((x - mean) ** 2 for x in col) / len(col)
            std = var ** 0.5
            if std == 0:
                norm_cols.append([0.0] * len(col))
            else:
                norm_cols.append([(x - mean) / std for x in col])
    return [list(row) for row in zip(*norm_cols)]


def _pareto_front(vectors: list[list[float]]) -> list[int]:
    front = []
    for i, v in enumerate(vectors):
        dominated = False
        for j, w in enumerate(vectors):
            if i == j:
                continue
            if all(w[k] >= v[k] for k in range(len(v))) and any(w[k] > v[k] for k in range(len(v))):
                dominated = True
                break
        if not dominated:
            front.append(i)
    return front


class MultiObjectiveCandidateSelector(CandidateSelector):
    def __init__(
        self,
        rng: random.Random | None,
        objectives: list[str] | None,
        selection_strategy: str,
        normalize: str,
        global_bonus: int,
        instance_sample_size: int | None,
    ):
        self.rng = rng or random.Random(0)
        self.objectives = objectives
        self.selection_strategy = selection_strategy
        self.normalize = normalize
        self.global_bonus = global_bonus
        self.instance_sample_size = instance_sample_size
        self.pareto_selector = ParetoCandidateSelector(self.rng)

    def _select_objective_pareto(self, state: GEPAState) -> int:
        objectives = self.objectives or list(state.prog_candidate_val_subscores[0][0].keys())  # type: ignore
        vectors = [
            [
                sum(s.get(obj, 0.0) for s in scores) / len(scores)  # type: ignore[union-attr]
                for obj in objectives
            ]
            for scores in state.prog_candidate_val_subscores
        ]
        vectors = _normalize_matrix(vectors, self.normalize)
        front = _pareto_front(vectors)
        return self.rng.choice(front)

    def _select_hybrid(self, state: GEPAState) -> int:
        objectives = self.objectives or list(state.prog_candidate_val_subscores[0][0].keys())  # type: ignore
        num_candidates = len(state.prog_candidate_val_subscores)
        num_instances = len(state.prog_candidate_val_subscores[0])
        inst_indices = list(range(num_instances))
        if self.instance_sample_size is not None and self.instance_sample_size < num_instances:
            inst_indices = self.rng.sample(inst_indices, self.instance_sample_size)

        weights = [0 for _ in range(num_candidates)]
        for inst in inst_indices:
            vectors = [
                [
                    state.prog_candidate_val_subscores[cand][inst].get(obj, 0.0)  # type: ignore[union-attr]
                    for obj in objectives
                ]
                for cand in range(num_candidates)
            ]
            vectors = _normalize_matrix(vectors, self.normalize)
            front = _pareto_front(vectors)
            for idx in front:
                weights[idx] += 1

        # Global objective means
        vectors = [
            [
                sum(s.get(obj, 0.0) for s in scores) / len(scores)  # type: ignore[union-attr]
                for obj in objectives
            ]
            for scores in state.prog_candidate_val_subscores
        ]
        vectors = _normalize_matrix(vectors, self.normalize)
        front = _pareto_front(vectors)
        for idx in front:
            weights[idx] += self.global_bonus

        sampling_list = [i for i, w in enumerate(weights) for _ in range(max(w, 0))]
        assert sampling_list, "No candidates to sample"
        return self.rng.choice(sampling_list)

    def select_candidate_idx(self, state: GEPAState) -> int:
        if not state.is_multi_objective:
            return self.pareto_selector.select_candidate_idx(state)

        strategy = self.selection_strategy
        if strategy == "auto":
            strategy = "objective_pareto"
        if strategy == "objective_pareto":
            return self._select_objective_pareto(state)
        if strategy == "hybrid":
            return self._select_hybrid(state)
        if strategy == "instance_pareto":
            return self.pareto_selector.select_candidate_idx(state)
        raise ValueError(f"Unknown selection strategy {self.selection_strategy}")

class CurrentBestCandidateSelector(CandidateSelector):
    def __init__(self):
        pass

    def select_candidate_idx(self, state: GEPAState) -> int:
        assert len(state.per_program_tracked_scores) == len(state.program_candidates)
        return idxmax(state.per_program_tracked_scores)
