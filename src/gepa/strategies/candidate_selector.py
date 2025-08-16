# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa
import random
from collections import defaultdict
from typing import Iterable

from gepa.core.state import GEPAState
from gepa.gepa_utils import idxmax, select_program_candidate_from_pareto_front
from gepa.proposer.reflective_mutation.base import CandidateSelector


class ParetoCandidateSelector(CandidateSelector):
    def __init__(
        self,
        rng: random.Random | None,
        selection_strategy: str = "instance_pareto",
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

    # --------- helpers ---------
    def _resolve_strategy(self, state: GEPAState) -> str:
        if self.selection_strategy == "auto":
            return "objective_pareto" if getattr(state, "is_multi_objective", False) else "instance_pareto"
        return self.selection_strategy

    def _aggregate_objective_scores(self, state: GEPAState):
        assert state.prog_candidate_val_subscores
        # Determine objective names
        if self.objectives is None:
            first = state.prog_candidate_val_subscores[0][0]
            if isinstance(first, dict):
                objectives = list(first.keys())
            else:
                objectives = ["score"]
        else:
            objectives = self.objectives
        vectors = []
        for cand_scores in state.prog_candidate_val_subscores:
            totals = defaultdict(float)
            counts = defaultdict(int)
            for score in cand_scores:
                if isinstance(score, dict):
                    for obj in objectives:
                        if obj in score:
                            totals[obj] += score[obj]
                            counts[obj] += 1
                else:
                    totals[objectives[0]] += score
                    counts[objectives[0]] += 1
            means = {obj: (totals[obj] / counts[obj] if counts[obj] else 0.0) for obj in objectives}
            vectors.append(means)
        return objectives, vectors

    def _normalize_vectors(self, objectives, vectors):
        norm_vectors = [[0.0 for _ in objectives] for _ in vectors]
        for j, obj in enumerate(objectives):
            vals = [vec[obj] for vec in vectors]
            if self.normalize == "minmax":
                mn, mx = min(vals), max(vals)
                rng = mx - mn if mx - mn != 0 else 1.0
                for i, v in enumerate(vals):
                    norm_vectors[i][j] = (v - mn) / rng
            else:  # zscore
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / len(vals)
                std = var ** 0.5 if var != 0 else 1.0
                for i, v in enumerate(vals):
                    norm_vectors[i][j] = (v - mean) / std
        return norm_vectors

    def _objective_front(self, state: GEPAState) -> set[int]:
        objectives, vectors = self._aggregate_objective_scores(state)
        norm_vectors = self._normalize_vectors(objectives, vectors)
        m = len(objectives)
        front = set(range(len(norm_vectors)))
        for i, v in enumerate(norm_vectors):
            for j, u in enumerate(norm_vectors):
                if i == j:
                    continue
                if all(u[k] >= v[k] for k in range(m)) and any(u[k] > v[k] for k in range(m)):
                    front.discard(i)
                    break
        return front

    def _instance_weights(self, state: GEPAState):
        fronts = state.program_at_pareto_front_valset
        if self.instance_sample_size is not None and len(fronts) > self.instance_sample_size:
            fronts = self.rng.sample(fronts, self.instance_sample_size)
        freq: dict[int, int] = defaultdict(int)
        for front in remove_empty(fronts):
            for idx in front:
                freq[idx] += 1
        return freq

    def _compute_weights(self, state: GEPAState, strategy: str):
        if strategy == "instance_pareto":
            return self._instance_weights(state)
        if strategy == "objective_pareto":
            front = self._objective_front(state)
            return {idx: 1 for idx in front}
        if strategy == "hybrid":
            weights = self._instance_weights(state)
            front = self._objective_front(state)
            for idx in front:
                weights[idx] = weights.get(idx, 0) + self.global_bonus
            return weights
        raise ValueError(f"Unknown selection strategy: {strategy}")

    # --------- public API ---------
    def select_candidate_idx(self, state: GEPAState) -> int:
        assert len(state.per_program_tracked_scores) == len(state.program_candidates)
        strategy = self._resolve_strategy(state)
        if strategy == "instance_pareto":
            return select_program_candidate_from_pareto_front(
                state.program_at_pareto_front_valset,
                state.per_program_tracked_scores,
                self.rng,
                instance_sample_size=self.instance_sample_size,
            )
        weights = self._compute_weights(state, strategy)
        sampling_list = [idx for idx, w in weights.items() for _ in range(w)]
        assert sampling_list, "No candidates to sample from"
        return self.rng.choice(sampling_list)


class CurrentBestCandidateSelector(CandidateSelector):
    def __init__(self):
        pass

    def select_candidate_idx(self, state: GEPAState) -> int:
        assert len(state.per_program_tracked_scores) == len(state.program_candidates)
        return idxmax(state.per_program_tracked_scores)


def remove_empty(fronts: Iterable[set[int]]):
    return [front for front in fronts if front]
