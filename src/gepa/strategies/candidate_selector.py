# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import random

from gepa.core.state import GEPAState
from gepa.gepa_utils import idxmax, select_program_candidate_from_pareto_front
from gepa.proposer.reflective_mutation.base import CandidateSelector


class ParetoCandidateSelector(CandidateSelector):
    def __init__(
        self,
        rng: random.Random | None,
        objectives: list[str] | None = None,
        selection_strategy: str = "auto",
        normalize: str = "zscore",
        global_bonus: int = 3,
        instance_sample_size: int | None = None,
    ):
        self.rng = rng or random.Random(0)
        self.objectives = objectives
        self.selection_strategy = selection_strategy
        self.normalize = normalize
        self.global_bonus = global_bonus
        self.instance_sample_size = instance_sample_size

    def _normalize(self, matrix: list[list[float]]) -> list[list[float]]:
        if not matrix:
            return matrix
        cols = list(zip(*matrix))
        norm_cols: list[list[float]] = []
        for col in cols:
            if self.normalize == "minmax":
                cmin, cmax = min(col), max(col)
                if cmax > cmin:
                    norm_cols.append([(v - cmin) / (cmax - cmin) for v in col])
                else:
                    norm_cols.append([0.0 for _ in col])
            else:  # zscore
                mean = sum(col) / len(col)
                var = sum((v - mean) ** 2 for v in col) / len(col)
                std = var ** 0.5
                if std > 0:
                    norm_cols.append([(v - mean) / std for v in col])
                else:
                    norm_cols.append([0.0 for _ in col])
        return [list(row) for row in zip(*norm_cols)]

    @staticmethod
    def _nondominated(matrix: list[list[float]]) -> list[int]:
        dominated: set[int] = set()
        for i, a in enumerate(matrix):
            if i in dominated:
                continue
            for j, b in enumerate(matrix):
                if i == j or j in dominated:
                    continue
                if all(b[k] >= a[k] for k in range(len(a))) and any(b[k] > a[k] for k in range(len(a))):
                    dominated.add(i)
                    break
        return [i for i in range(len(matrix)) if i not in dominated]

    def _objective_front(self, state: GEPAState) -> list[int]:
        assert state.program_objective_scores is not None
        objectives = self.objectives or state.objectives or list(state.program_objective_scores[0].keys())
        matrix = [[scores[obj] for obj in objectives] for scores in state.program_objective_scores]
        matrix = self._normalize(matrix)
        return self._nondominated(matrix)

    def _sample_from_instance_front(self, fronts, scores) -> int:
        return select_program_candidate_from_pareto_front(fronts, scores, self.rng)

    def select_candidate_idx(self, state: GEPAState) -> int:
        strategy = self.selection_strategy
        if strategy == "auto":
            strategy = "objective_pareto" if state.is_multi_objective else "instance_pareto"
        if strategy in {"objective_pareto", "hybrid"} and (state.objectives is None or len(state.objectives) <= 1):
            strategy = "instance_pareto"

        if strategy == "objective_pareto":
            front = self._objective_front(state)
            return self.rng.choice(front)

        fronts = state.program_at_pareto_front_valset
        if self.instance_sample_size is not None and len(fronts) > self.instance_sample_size:
            idxs = self.rng.sample(range(len(fronts)), self.instance_sample_size)
            fronts = [fronts[i] for i in idxs]

        if strategy == "instance_pareto":
            return self._sample_from_instance_front(fronts, state.per_program_tracked_scores)

        if strategy == "hybrid":
            weights: dict[int, float] = {}
            for front in fronts:
                for prog in front:
                    weights[prog] = weights.get(prog, 0) + 1
            obj_front = self._objective_front(state)
            for prog in obj_front:
                weights[prog] = weights.get(prog, 0) + self.global_bonus
            sampling_list = [p for p, w in weights.items() for _ in range(int(w))]
            return self.rng.choice(sampling_list)

        raise ValueError(f"Unknown selection_strategy: {strategy}")

class CurrentBestCandidateSelector(CandidateSelector):
    def __init__(self):
        pass

    def select_candidate_idx(self, state: GEPAState) -> int:
        assert len(state.per_program_tracked_scores) == len(state.program_candidates)
        return idxmax(state.per_program_tracked_scores)
