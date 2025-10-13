# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Generic

from gepa.core.adapter import RolloutOutput
from gepa.core.state import ProgramIdx, ValId, ValScores


@dataclass(frozen=True)
class GEPAResult(Generic[RolloutOutput]):
    """
    Immutable snapshot of a GEPA run with convenience accessors.

    - candidates: list of proposed candidates (component_name -> component_text)
    - parents: lineage info; for each candidate i, parents[i] is a list of parent indices or None
    - val_aggregate_scores: per-candidate aggregate score on the validation set (higher is better)
    - val_subscores: per-candidate per-instance scores on the validation set (len == num_val_instances)
    - per_val_instance_best_candidates: for each tracked frontier dimension (validation instance or objective), a set of
      candidate indices achieving the current best score on that dimension
    - discovery_eval_counts: number of metric calls accumulated up to the discovery of each candidate

    Optional fields:
    - best_outputs_valset: per-task best outputs on the validation set. [task_idx -> [(program_idx_1, output_1), (program_idx_2, output_2), ...]]

    Run-level metadata:
    - total_metric_calls: total number of metric calls made across the run
    - num_full_val_evals: number of full validation evaluations performed
    - run_dir: where artifacts were written (if any)
    - seed: RNG seed for reproducibility (if known)
    - tracked_scores: optional tracked aggregate scores (if different from val_aggregate_scores)

    Convenience:
    - best_idx: candidate index with the highest val_aggregate_scores
    - best_candidate: the program text mapping for best_idx
    - non_dominated_indices(): candidate indices that are not dominated across per-instance pareto fronts
    - lineage(idx): parent chain from base to idx
    - diff(parent_idx, child_idx, only_changed=True): component-wise diff between two candidates
    - best_k(k): top-k candidates by aggregate val score
    - instance_winners(t): set of candidates on the pareto front for val instance t
    - to_dict(...), save_json(...): serialization helpers
    """

    # Core data
    candidates: list[dict[str, str]]
    parents: list[list[ProgramIdx | None]]
    val_aggregate_scores: list[float]
    val_subscores: list[ValScores]
    per_val_instance_best_candidates: dict[ValId, set[ProgramIdx]]
    discovery_eval_counts: list[int]

    frontier_type: str = "instance"
    frontier_dimension_labels: list[str] | None = None
    objective_scores: list[dict[str, float]] | None = None
    frontier_scores: dict[str, float] | None = None
    frontier_programs: dict[str, set[int]] | None = None

    # Optional data
    best_outputs_valset: dict[ValId, list[tuple[ProgramIdx, RolloutOutput]]] | None = None

    # Run metadata (optional)
    total_metric_calls: int | None = None
    num_full_val_evals: int | None = None
    run_dir: str | None = None
    seed: int | None = None

    # -------- Convenience properties --------
    @property
    def num_candidates(self) -> int:
        return len(self.candidates)

    @property
    def num_val_instances(self) -> int:
        return len(self.per_val_instance_best_candidates)

    @property
    def best_idx(self) -> int:
        scores = self.val_aggregate_scores
        return max(range(len(scores)), key=lambda i: scores[i])

    @property
    def best_candidate(self) -> dict[str, str]:
        return self.candidates[self.best_idx]

    def to_dict(self) -> dict[str, Any]:
        cands = [dict(cand.items()) for cand in self.candidates]

        return dict(
            candidates=cands,
            parents=self.parents,
            val_aggregate_scores=self.val_aggregate_scores,
            val_subscores=self.val_subscores,
            frontier_type=self.frontier_type,
            frontier_dimension_labels=self.frontier_dimension_labels,
            objective_scores=self.objective_scores,
            frontier_scores=self.frontier_scores,
            frontier_programs={k: list(v) for k, v in self.frontier_programs.items()}
            if self.frontier_programs
            else None,
            best_outputs_valset=self.best_outputs_valset,
            per_val_instance_best_candidates={k: list(v) for k, v in self.per_val_instance_best_candidates.items()},
            discovery_eval_counts=self.discovery_eval_counts,
            total_metric_calls=self.total_metric_calls,
            num_full_val_evals=self.num_full_val_evals,
            run_dir=self.run_dir,
            seed=self.seed,
            best_idx=self.best_idx,
        )

    @staticmethod
    def from_state(state: Any, run_dir: str | None = None, seed: int | None = None) -> "GEPAResult":
        """
        Build a GEPAResult from a GEPAState.
        """
        return GEPAResult(
            candidates=list(state.program_candidates),
            parents=list(state.parent_program_for_candidate),
            val_aggregate_scores=list(state.program_full_scores_val_set),
            best_outputs_valset=getattr(state, "best_outputs_valset", None),
            val_subscores=[dict(scores) for scores in state.prog_candidate_val_subscores],
            per_val_instance_best_candidates={
                val_id: set(front) for val_id, front in state.program_at_pareto_front_valset.items()
            },
            discovery_eval_counts=list(state.num_metric_calls_by_discovery),
            frontier_type=getattr(state, "frontier_type", "instance"),
            frontier_dimension_labels=list(
                getattr(
                    state,
                    "frontier_dimension_labels",
                    [f"instance:{idx}" for idx in range(len(state.program_at_pareto_front_valset))],
                )
            ),
            objective_scores=[
                dict(scores)
                for scores in getattr(
                    state,
                    "program_objective_scores",
                    [{} for _ in state.program_candidates],
                )
            ],
            frontier_scores=(
                dict(frontier_scores_attr)
                if isinstance(frontier_scores_attr := getattr(state, "frontier_scores", None), Mapping)
                else None
            ),
            frontier_programs=(
                {label: set(progs) for label, progs in frontier_programs_attr.items()}
                if isinstance(frontier_programs_attr := getattr(state, "frontier_programs", None), Mapping)
                else None
            ),
            total_metric_calls=getattr(state, "total_num_evals", None),
            num_full_val_evals=getattr(state, "num_full_ds_evals", None),
            run_dir=run_dir,
            seed=seed,
        )
