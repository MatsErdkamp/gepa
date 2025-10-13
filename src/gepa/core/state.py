# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Callable, ClassVar, Generic, Hashable, Literal, TypeAlias, TypeVar

from gepa.core.adapter import RolloutOutput
from gepa.gepa_utils import json_default

# Types for GEPAState
ProgramIdx = int
ValId = TypeVar("ValId", bound=Hashable)
"""Opaque identifier for valset examples"""
ValScores: TypeAlias = dict[ValId, float]
ValOutputs: TypeAlias = dict[ValId, RolloutOutput]
FrontierType = Literal["instance", "objective", "hybrid"]


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def unpack_evaluation_output(
    eval_output: tuple[Any, ...] | list[Any],
) -> tuple[list[RolloutOutput], list[float], Any]:
    if not isinstance(eval_output, (tuple, list)):
        raise TypeError("Evaluation output must be a tuple or list")

    if len(eval_output) == 2:
        outputs, instance_scores = eval_output
        subscores = None
    elif len(eval_output) == 3:
        outputs, instance_scores, subscores = eval_output
    else:
        raise ValueError("Evaluation output must be a tuple of length 2 or 3")

    outputs_list = list(outputs) if outputs is not None else []
    return outputs_list, list(instance_scores), subscores


def aggregate_objective_scores(subscores: Any) -> dict[str, float]:
    if subscores is None:
        return {}

    if isinstance(subscores, Mapping):
        return {str(k): float(v) for k, v in subscores.items()}

    if not _is_sequence(subscores):
        raise TypeError("Objective subscores must be a mapping or a sequence")

    aggregated: dict[str, float] = {}
    count = 0
    for entry in subscores:
        if entry is None:
            continue

        count += 1
        if isinstance(entry, Mapping):
            for key, value in entry.items():
                aggregated[str(key)] = aggregated.get(str(key), 0.0) + float(value)
        elif _is_sequence(entry):
            for idx, value in enumerate(entry):
                key = str(idx)
                aggregated[key] = aggregated.get(key, 0.0) + float(value)
        else:
            aggregated["0"] = aggregated.get("0", 0.0) + float(entry)

    if count == 0:
        return {}

    return {key: value / count for key, value in aggregated.items()}


def compute_frontier_dimensions(
    frontier_type: FrontierType,
    instance_scores: Mapping[ValId, float] | Sequence[float],
    objective_scores: Mapping[str, float] | None,
) -> tuple[list[str], list[float]]:
    labels: list[str] = []
    scores: list[float] = []

    if frontier_type not in {"instance", "objective", "hybrid"}:
        raise ValueError(f"Unknown frontier_type: {frontier_type}")

    if frontier_type in {"objective", "hybrid"}:
        if not objective_scores:
            raise ValueError("Objective frontier requested but no objective subscores were provided")
        for key in sorted(objective_scores):
            labels.append(f"objective:{key}")
            scores.append(float(objective_scores[key]))

    if frontier_type in {"instance", "hybrid"}:
        if isinstance(instance_scores, Mapping):
            items = instance_scores.items()
        else:
            items = enumerate(instance_scores)
        for key, value in items:
            labels.append(f"instance:{key}")
            scores.append(float(value))

    return labels, scores


class GEPAState(Generic[RolloutOutput, ValId]):
    _VALIDATION_SCHEMA_VERSION: ClassVar[int] = 2

    program_candidates: list[dict[str, str]]
    parent_program_for_candidate: list[list[ProgramIdx | None]]

    prog_candidate_val_subscores: list[ValScores]
    program_objective_scores: list[dict[str, float]]

    pareto_front_valset: ValScores
    program_at_pareto_front_valset: dict[ValId, set[ProgramIdx]]

    frontier_type: FrontierType
    frontier_dimension_labels: list[str]
    frontier_scores: dict[str, float]
    frontier_programs: dict[str, set[ProgramIdx]]

    list_of_named_predictors: list[str]
    named_predictor_id_to_update_next_for_program_candidate: list[int]

    i: int
    num_full_ds_evals: int

    total_num_evals: int

    num_metric_calls_by_discovery: list[int]

    full_program_trace: list
    best_outputs_valset: dict[ValId, list[tuple[ProgramIdx, RolloutOutput]]] | None = None

    validation_schema_version: int
    num_val_instances: int

    frontier_type: FrontierType
    frontier_dimension_labels: list[str]
    num_val_instances: int

    def __init__(
        self,
        seed_candidate: dict[str, str],
        base_valset_eval_output: tuple[ValOutputs, ValScores],
        track_best_outputs: bool = False,
        frontier_type: FrontierType = "instance",
        frontier_dimension_labels: list[str] | None = None,
        base_frontier_scores: list[float] | None = None,
        objective_scores: dict[str, float] | None = None,
    ):
        base_outputs, base_scores = base_valset_eval_output
        self.program_candidates = [seed_candidate]
        self.prog_candidate_val_subscores = [dict(base_scores)]
        self.program_objective_scores = [dict(objective_scores or {})]

        self.pareto_front_valset = {val_id: score for val_id, score in base_scores.items()}
        self.parent_program_for_candidate = [[None]]
        self.program_at_pareto_front_valset = {val_id: {0} for val_id in base_scores.keys()}

        self.list_of_named_predictors = list(seed_candidate.keys())
        self.named_predictor_id_to_update_next_for_program_candidate = [0]
        self.i = -1

        self.num_metric_calls_by_discovery = [0]

        self.best_outputs_valset = (
            {val_id: [(0, output)] for val_id, output in base_outputs.items()} if track_best_outputs else None
        )

        self.full_program_trace = []
        self.validation_schema_version = self._VALIDATION_SCHEMA_VERSION

        self.num_full_ds_evals = 0
        self.total_num_evals = 0
        self.num_val_instances = len(base_scores)

        self.frontier_type = frontier_type
        if frontier_dimension_labels is None:
            self.frontier_dimension_labels = [f"instance:{val_id}" for val_id in base_scores.keys()]
        else:
            self.frontier_dimension_labels = list(frontier_dimension_labels)

        if base_frontier_scores is None:
            # Default: align with per-instance scores in insertion order
            base_frontier_scores = [base_scores[val_id] for val_id in base_scores.keys()]

        self.frontier_scores = {
            label: float(score)
            for label, score in zip(self.frontier_dimension_labels, base_frontier_scores, strict=False)
        }
        self.frontier_programs = {label: {0} for label in self.frontier_scores.keys()}

    def is_consistent(self):
        assert len(self.program_candidates) == len(self.parent_program_for_candidate)
        assert len(self.program_candidates) == len(self.named_predictor_id_to_update_next_for_program_candidate)
        assert len(self.program_candidates) == len(self.prog_candidate_val_subscores)
        assert len(self.program_candidates) == len(self.num_metric_calls_by_discovery)
        assert len(self.program_candidates) == len(self.program_objective_scores)

        for front in self.program_at_pareto_front_valset.values():
            for prog_idx in front:
                assert prog_idx < len(self.program_candidates), (
                    "Program index in valset pareto front exceeds number of program candidates"
                )

        assert set(self.pareto_front_valset.keys()) == set(self.program_at_pareto_front_valset.keys())

        for label, progs in self.frontier_programs.items():
            assert label in self.frontier_scores
            for prog_idx in progs:
                assert prog_idx < len(self.program_candidates), (
                    "Program index in frontier programs exceeds number of program candidates"
                )

        return True

    def save(self, run_dir: str | None):
        if run_dir is None:
            return
        with open(os.path.join(run_dir, "gepa_state.bin"), "wb") as f:
            import pickle

            d = dict(self.__dict__.items())
            d["validation_schema_version"] = GEPAState._VALIDATION_SCHEMA_VERSION
            pickle.dump(d, f)

    @staticmethod
    def load(run_dir: str) -> "GEPAState":
        with open(os.path.join(run_dir, "gepa_state.bin"), "rb") as f:
            import pickle

            d = pickle.load(f)

        version = d.get("validation_schema_version")
        if version is None or version < GEPAState._VALIDATION_SCHEMA_VERSION:
            GEPAState._migrate_legacy_state_dict(d)

        state = GEPAState.__new__(GEPAState)
        state.__dict__.update(d)

        state.validation_schema_version = GEPAState._VALIDATION_SCHEMA_VERSION
        state.frontier_type = getattr(state, "frontier_type", "instance")
        state.frontier_dimension_labels = list(getattr(state, "frontier_dimension_labels", []))
        state.frontier_scores = dict(getattr(state, "frontier_scores", {}))
        state.frontier_programs = {
            label: set(progs) for label, progs in getattr(state, "frontier_programs", {}).items()
        }
        state.program_objective_scores = [
            dict(scores)
            for scores in getattr(state, "program_objective_scores", [{} for _ in state.program_candidates])
        ]
        state.num_val_instances = getattr(state, "num_val_instances", len(state.pareto_front_valset))

        assert set(state.pareto_front_valset.keys()) == set(state.program_at_pareto_front_valset.keys())
        assert len(state.program_candidates) == len(state.prog_candidate_val_subscores)
        assert len(state.program_candidates) == len(state.num_metric_calls_by_discovery)
        assert len(state.program_candidates) == len(state.parent_program_for_candidate)
        assert len(state.program_candidates) == len(state.named_predictor_id_to_update_next_for_program_candidate)
        assert len(state.program_candidates) == len(state.program_objective_scores)
        return state

    @staticmethod
    def _migrate_legacy_state_dict(d: dict[str, Any]) -> None:
        legacy_scores: list[list[float]] = d.pop("prog_candidate_val_subscores", [])
        d["prog_candidate_val_subscores"] = [
            {idx: score for idx, score in enumerate(scores)} for scores in legacy_scores
        ]

        if "pareto_front_valset" in d and isinstance(d["pareto_front_valset"], list):
            d["pareto_front_valset"] = {idx: score for idx, score in enumerate(d["pareto_front_valset"])}

        program_at_front = d.get("program_at_pareto_front_valset")
        if program_at_front and isinstance(program_at_front, list):
            d["program_at_pareto_front_valset"] = {idx: set(front) for idx, front in enumerate(program_at_front)}

        if "frontier_dimension_labels" not in d:
            d["frontier_dimension_labels"] = [f"instance:{idx}" for idx in d["pareto_front_valset"].keys()]
        if "frontier_scores" not in d:
            d["frontier_scores"] = {
                label: float(d["pareto_front_valset"].get(idx, float("-inf")))
                for label, idx in zip(d["frontier_dimension_labels"], d["pareto_front_valset"].keys(), strict=False)
            }
        if "frontier_programs" not in d:
            d["frontier_programs"] = {
                label: set(front)
                for label, front in zip(
                    d["frontier_dimension_labels"], d["program_at_pareto_front_valset"].values(), strict=False
                )
            }
        if "program_objective_scores" not in d:
            d["program_objective_scores"] = [{} for _ in d["prog_candidate_val_subscores"]]
        if "frontier_type" not in d:
            d["frontier_type"] = "instance"
        if "num_val_instances" not in d:
            d["num_val_instances"] = len(d["pareto_front_valset"])

    def get_program_average(self, program_idx: ProgramIdx) -> tuple[float, int]:
        val_scores = self.prog_candidate_val_subscores[program_idx]
        coverage = len(val_scores)
        if coverage == 0:
            return float("-inf"), 0
        return sum(val_scores.values()) / coverage, coverage

    @property
    def program_full_scores_val_set(self) -> list[float]:
        return [
            self.get_program_average(program_idx)[0] for program_idx in range(len(self.prog_candidate_val_subscores))
        ]

    @property
    def per_program_tracked_scores(self) -> list[float]:
        # NOTE(aria42): This same as valset program average scores, but this was already the case
        return [
            self.get_program_average(program_idx)[0] for program_idx in range(len(self.prog_candidate_val_subscores))
        ]

    @property
    def valset_evaluations(self) -> dict[ValId, list[ProgramIdx]]:
        result: dict[ValId, list[ProgramIdx]] = defaultdict(list)
        for program_idx, val_scores in enumerate(self.prog_candidate_val_subscores):
            for val_id in val_scores.keys():
                result[val_id].append(program_idx)
        return result

    def _update_pareto_front_for_val_id(
        self,
        val_id: ValId,
        score: float,
        program_idx: ProgramIdx,
        outputs: ValOutputs | None,
        run_dir: str | None,
        iteration: int,
    ) -> None:
        prev_score = self.pareto_front_valset.get(val_id, float("-inf"))
        if score > prev_score:
            self.pareto_front_valset[val_id] = score
            self.program_at_pareto_front_valset[val_id] = {program_idx}
            output = outputs.get(val_id) if outputs is not None else None
            if self.best_outputs_valset is not None and output is not None:
                self.best_outputs_valset[val_id] = [(program_idx, output)]
                if run_dir is not None:
                    task_dir = os.path.join(run_dir, "generated_best_outputs_valset", f"task_{val_id}")
                    os.makedirs(task_dir, exist_ok=True)
                    with open(os.path.join(task_dir, f"iter_{iteration}_prog_{program_idx}.json"), "w") as fout:
                        json.dump(output, fout, indent=4, default=json_default)
        elif score == prev_score:
            pareto_front = self.program_at_pareto_front_valset.setdefault(val_id, set())
            pareto_front.add(program_idx)
            output = outputs.get(val_id) if outputs is not None else None
            if self.best_outputs_valset is not None and output is not None:
                self.best_outputs_valset[val_id].append((program_idx, output))

    def _update_frontier_dimensions(
        self,
        new_program_idx: ProgramIdx,
        frontier_dimension_labels: list[str] | None,
        frontier_scores: list[float] | None,
    ) -> None:
        if not frontier_dimension_labels or frontier_scores is None:
            return

        if not self.frontier_dimension_labels:
            self.frontier_dimension_labels = list(frontier_dimension_labels)

        if len(frontier_dimension_labels) != len(frontier_scores):
            raise ValueError("Mismatch between frontier labels and scores length")

        for label, score in zip(frontier_dimension_labels, frontier_scores, strict=False):
            current_best = self.frontier_scores.get(label, float("-inf"))
            if score > current_best:
                self.frontier_scores[label] = float(score)
                self.frontier_programs[label] = {new_program_idx}
            elif score == current_best:
                self.frontier_programs.setdefault(label, set()).add(new_program_idx)

    def update_state_with_new_program(
        self,
        parent_program_idx: list[ProgramIdx],
        new_program: dict[str, str],
        valset_subscores: ValScores,
        valset_outputs: ValOutputs | None,
        run_dir: str | None,
        num_metric_calls_by_discovery_of_new_program: int,
        *,
        valset_score: float | None = None,
        frontier_dimension_labels: list[str] | None = None,
        frontier_scores: list[float] | None = None,
        objective_scores: dict[str, float] | None = None,
    ) -> tuple[ProgramIdx, ProgramIdx]:
        new_program_idx = len(self.program_candidates)
        self.program_candidates.append(new_program)
        self.num_metric_calls_by_discovery.append(num_metric_calls_by_discovery_of_new_program)

        max_predictor_id = max(
            [self.named_predictor_id_to_update_next_for_program_candidate[p] for p in parent_program_idx],
            default=0,
        )
        self.named_predictor_id_to_update_next_for_program_candidate.append(max_predictor_id)
        self.parent_program_for_candidate.append(list(parent_program_idx))

        self.prog_candidate_val_subscores.append(dict(valset_subscores))
        self.program_objective_scores.append(dict(objective_scores or {}))

        for val_id, score in valset_subscores.items():
            self._update_pareto_front_for_val_id(val_id, score, new_program_idx, valset_outputs, run_dir, self.i + 1)

        self._update_frontier_dimensions(new_program_idx, frontier_dimension_labels, frontier_scores)

        linear_pareto_front_program_idx = self._best_program_idx()
        return new_program_idx, linear_pareto_front_program_idx

    def _best_program_idx(self) -> ProgramIdx:
        best_idx = 0
        best_avg, best_cov = self.get_program_average(0)
        for idx in range(1, len(self.program_candidates)):
            avg, cov = self.get_program_average(idx)
            if avg > best_avg or (avg == best_avg and cov > best_cov):
                best_idx = idx
                best_avg, best_cov = avg, cov
        return best_idx

    def frontier_snapshot(self) -> dict[str, float]:
        return dict(self.frontier_scores)


def write_eval_scores_to_directory(scores: ValScores, output_dir: str):
    for val_id, score in scores.items():
        task_dir = os.path.join(output_dir, f"task_{val_id}")
        os.makedirs(task_dir, exist_ok=True)
        with open(os.path.join(task_dir, f"iter_{0}_prog_0.json"), "w") as f:
            json.dump(score, f, indent=4, default=json_default)


def initialize_gepa_state(
    run_dir: str | None,
    logger,
    seed_candidate: dict[str, str],
    valset_evaluator: Callable[[dict[str, str]], tuple[ValOutputs, ValScores, Any]],
    track_best_outputs: bool = False,
    frontier_type: FrontierType = "instance",
):
    if run_dir is not None and os.path.exists(os.path.join(run_dir, "gepa_state.bin")):
        logger.log("Loading gepa state from run dir")
        gepa_state = GEPAState.load(run_dir)
        if gepa_state.frontier_type != frontier_type:
            raise ValueError(
                "frontier_type mismatch when resuming from run_dir. "
                f"Expected '{gepa_state.frontier_type}', received '{frontier_type}'."
            )
    else:
        num_evals_run = 0

        valset_eval = valset_evaluator(seed_candidate)
        if not isinstance(valset_eval, (tuple, list)):
            raise TypeError("valset_evaluator must return a tuple or list")
        if len(valset_eval) == 2:
            seed_val_outputs, seed_val_scores = valset_eval
            seed_subscores = None
        elif len(valset_eval) == 3:
            seed_val_outputs, seed_val_scores, seed_subscores = valset_eval
        else:
            raise ValueError("valset_evaluator must return 2 or 3 values")

        seed_val_outputs = dict(seed_val_outputs)
        seed_val_scores = dict(seed_val_scores)
        objective_scores = aggregate_objective_scores(seed_subscores)
        frontier_labels, frontier_scores = compute_frontier_dimensions(
            frontier_type,
            seed_val_scores,
            objective_scores,
        )

        if run_dir is not None:
            write_eval_scores_to_directory(seed_val_scores, os.path.join(run_dir, "generated_best_outputs_valset"))
        num_evals_run += len(seed_val_scores)

        gepa_state = GEPAState(
            seed_candidate,
            (seed_val_outputs, seed_val_scores),
            track_best_outputs=track_best_outputs,
            frontier_type=frontier_type,
            frontier_dimension_labels=frontier_labels,
            base_frontier_scores=frontier_scores,
            objective_scores=objective_scores,
        )

        gepa_state.num_full_ds_evals = 1
        gepa_state.total_num_evals = num_evals_run

    return gepa_state
