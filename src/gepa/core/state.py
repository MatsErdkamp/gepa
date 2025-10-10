# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import json
import os
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Generic, Literal

from gepa.core.adapter import RolloutOutput
from gepa.gepa_utils import idxmax, json_default

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
    instance_scores: Sequence[float],
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
        for idx, value in enumerate(instance_scores):
            labels.append(f"instance:{idx}")
            scores.append(float(value))

    return labels, scores


class GEPAState(Generic[RolloutOutput]):
    program_candidates: list[dict[str, str]]
    parent_program_for_candidate: list[list[int | None]]

    program_full_scores_val_set: list[float]

    program_at_pareto_front_valset: list[set[int]]

    prog_candidate_val_subscores: list[list[float]]
    program_objective_scores: list[dict[str, float]]

    list_of_named_predictors: list[str]
    named_predictor_id_to_update_next_for_program_candidate: list[int]

    i: int
    num_full_ds_evals: int

    total_num_evals: int

    num_metric_calls_by_discovery: list[int]

    full_program_trace: list

    per_program_tracked_scores: list[float]

    best_outputs_valset: list[tuple[int, list[RolloutOutput]]] | None = None

    frontier_type: FrontierType
    frontier_dimension_labels: list[str]
    num_val_instances: int

    def __init__(
        self,
        seed_candidate: dict[str, str],
        base_outputs: list[RolloutOutput],
        base_instance_scores: list[float],
        frontier_type: FrontierType,
        frontier_dimension_labels: list[str],
        base_frontier_scores: list[float],
        objective_scores: dict[str, float] | None = None,
        track_best_outputs: bool = False,
    ):
        self.frontier_type = frontier_type
        self.frontier_dimension_labels = list(frontier_dimension_labels)
        if len(self.frontier_dimension_labels) != len(base_frontier_scores):
            raise ValueError("frontier_dimension_labels and base_frontier_scores must align")

        self.program_candidates = [seed_candidate]
        self.parent_program_for_candidate = [[None]]
        self.list_of_named_predictors = list(seed_candidate.keys())
        self.named_predictor_id_to_update_next_for_program_candidate = [0]
        self.i = -1

        self.num_val_instances = len(base_instance_scores)

        valset_base_score = sum(base_instance_scores) / len(base_instance_scores) if base_instance_scores else 0.0

        self.program_full_scores_val_set = [valset_base_score]
        self.per_program_tracked_scores = [valset_base_score]

        self.prog_candidate_val_subscores = [list(base_instance_scores)]
        self.program_objective_scores = [dict(objective_scores or {})]

        self.pareto_front_valset = list(base_frontier_scores)
        self.program_at_pareto_front_valset = [{0} for _ in range(len(base_frontier_scores))]

        self.num_metric_calls_by_discovery = [0]

        if track_best_outputs:
            self.best_outputs_valset = [[(0, output)] for output in base_outputs]

        self.full_program_trace = []

    def is_consistent(self):
        assert len(self.program_candidates) == len(self.program_full_scores_val_set)
        assert len(self.program_candidates) == len(self.per_program_tracked_scores)
        assert len(self.program_candidates) == len(self.parent_program_for_candidate)
        assert len(self.program_candidates) == len(self.named_predictor_id_to_update_next_for_program_candidate)
        assert len(self.program_candidates) == len(self.program_objective_scores)

        assert len(self.prog_candidate_val_subscores) == len(self.program_candidates)
        assert len(self.pareto_front_valset) == len(self.program_at_pareto_front_valset)
        assert len(self.pareto_front_valset) == len(self.frontier_dimension_labels)
        assert len(self.program_candidates) == len(self.num_metric_calls_by_discovery)

        if self.prog_candidate_val_subscores:
            assert all(len(subscores) == self.num_val_instances for subscores in self.prog_candidate_val_subscores), (
                "Mismatch between stored instance scores and num_val_instances"
            )

        for prog_list in self.program_at_pareto_front_valset:
            for prog_idx in prog_list:
                assert prog_idx < len(self.program_candidates), (
                    "Program index in valset pareto front exceeds number of program candidates"
                )

        return True

    def save(self, run_dir: str | None):
        if run_dir is None:
            return
        with open(os.path.join(run_dir, "gepa_state.bin"), "wb") as f:
            import pickle

            d = dict(self.__dict__.items())
            pickle.dump(d, f)

    @staticmethod
    def load(run_dir: str) -> "GEPAState":
        with open(os.path.join(run_dir, "gepa_state.bin"), "rb") as f:
            import pickle

            d = pickle.load(f)
        state = GEPAState.__new__(GEPAState)
        state.__dict__.update(d)

        if not hasattr(state, "frontier_type"):
            state.frontier_type = "instance"
        if not hasattr(state, "frontier_dimension_labels"):
            state.frontier_dimension_labels = [f"instance:{idx}" for idx in range(len(state.pareto_front_valset))]
        if not hasattr(state, "program_objective_scores"):
            state.program_objective_scores = [{} for _ in state.program_candidates]
        if not hasattr(state, "num_val_instances"):
            if state.prog_candidate_val_subscores:
                state.num_val_instances = len(state.prog_candidate_val_subscores[0])
            else:
                state.num_val_instances = len(state.program_at_pareto_front_valset)

        if len(state.frontier_dimension_labels) != len(state.pareto_front_valset):
            state.frontier_dimension_labels = [f"instance:{idx}" for idx in range(len(state.pareto_front_valset))]

        if len(state.program_objective_scores) != len(state.program_candidates):
            state.program_objective_scores = [
                state.program_objective_scores[idx] if idx < len(state.program_objective_scores) else {}
                for idx in range(len(state.program_candidates))
            ]

        assert len(state.program_candidates) == len(state.program_full_scores_val_set)
        assert len(state.pareto_front_valset) == len(state.program_at_pareto_front_valset)

        assert len(state.program_candidates) == len(state.parent_program_for_candidate)
        assert len(state.program_candidates) == len(state.named_predictor_id_to_update_next_for_program_candidate)

        return state

    def update_state_with_new_program(
        self,
        parent_program_idx: list[int],
        new_program: dict[str, str],
        valset_score: float,
        valset_outputs: Any,
        instance_scores: list[float],
        frontier_scores: list[float],
        frontier_dimension_labels: list[str],
        objective_scores: dict[str, float] | None,
        run_dir: str | None,
        num_metric_calls_by_discovery_of_new_program: int,
    ):
        new_program_idx = len(self.program_candidates)
        self.program_candidates.append(new_program)
        self.num_metric_calls_by_discovery.append(num_metric_calls_by_discovery_of_new_program)
        # Find the highest predictor id from the parent programs
        max_predictor_id = max(
            [self.named_predictor_id_to_update_next_for_program_candidate[p] for p in parent_program_idx]
        )
        self.named_predictor_id_to_update_next_for_program_candidate.append(max_predictor_id)
        self.parent_program_for_candidate.append(list(parent_program_idx))

        self.prog_candidate_val_subscores.append(list(instance_scores))
        self.program_objective_scores.append(dict(objective_scores or {}))
        self.program_full_scores_val_set.append(valset_score)
        if frontier_dimension_labels != self.frontier_dimension_labels:
            raise ValueError("Frontier dimension labels changed during the run")

        for dim_idx, (old_score, new_score) in enumerate(zip(self.pareto_front_valset, frontier_scores, strict=False)):
            label = self.frontier_dimension_labels[dim_idx]
            if new_score > old_score:
                self.pareto_front_valset[dim_idx] = new_score
                self.program_at_pareto_front_valset[dim_idx] = {new_program_idx}

                if label.startswith("instance:") and self.best_outputs_valset is not None:
                    instance_idx = int(label.split(":", 1)[1])
                    self.best_outputs_valset[instance_idx] = [(new_program_idx, valset_outputs[instance_idx])]

                    if run_dir is not None:
                        os.makedirs(
                            os.path.join(
                                run_dir,
                                "generated_best_outputs_valset",
                                f"task_{instance_idx}",
                            ),
                            exist_ok=True,
                        )
                        with open(
                            os.path.join(
                                run_dir,
                                "generated_best_outputs_valset",
                                f"task_{instance_idx}",
                                f"iter_{self.i + 1}_prog_{new_program_idx}.json",
                            ),
                            "w",
                        ) as f:
                            json.dump(valset_outputs[instance_idx], f, indent=4, default=json_default)
            elif new_score == old_score:
                self.program_at_pareto_front_valset[dim_idx].add(new_program_idx)
                if label.startswith("instance:") and self.best_outputs_valset is not None:
                    instance_idx = int(label.split(":", 1)[1])
                    self.best_outputs_valset[instance_idx].append((new_program_idx, valset_outputs[instance_idx]))

        assert len(frontier_scores) == len(self.program_at_pareto_front_valset)

        if self.best_outputs_valset is not None and self.frontier_type == "objective":
            for instance_idx, output in enumerate(valset_outputs):
                current_entries = self.best_outputs_valset[instance_idx]
                best_score = max(
                    self.prog_candidate_val_subscores[prog_idx][instance_idx] for prog_idx, _ in current_entries
                )
                new_score = instance_scores[instance_idx]
                if new_score > best_score:
                    self.best_outputs_valset[instance_idx] = [(new_program_idx, output)]
                    if run_dir is not None:
                        os.makedirs(
                            os.path.join(
                                run_dir,
                                "generated_best_outputs_valset",
                                f"task_{instance_idx}",
                            ),
                            exist_ok=True,
                        )
                        with open(
                            os.path.join(
                                run_dir,
                                "generated_best_outputs_valset",
                                f"task_{instance_idx}",
                                f"iter_{self.i + 1}_prog_{new_program_idx}.json",
                            ),
                            "w",
                        ) as f:
                            json.dump(output, f, indent=4, default=json_default)
                elif new_score == best_score:
                    self.best_outputs_valset[instance_idx].append((new_program_idx, output))

        self.per_program_tracked_scores = self.program_full_scores_val_set

        linear_pareto_front_program_idx = idxmax(self.per_program_tracked_scores)

        return new_program_idx, linear_pareto_front_program_idx


def write_eval_output_to_directory(
    scores: Sequence[float],
    output_dir: str,
):
    for task_idx, score in enumerate(scores):
        os.makedirs(os.path.join(output_dir, f"task_{task_idx}"), exist_ok=True)
        with open(
            os.path.join(output_dir, f"task_{task_idx}", f"iter_{0}_prog_0.json"),
            "w",
        ) as f:
            json.dump(score, f, indent=4, default=json_default)


def initialize_gepa_state(
    run_dir: str | None,
    logger,
    seed_candidate: dict[str, str],
    valset_evaluator: Callable[[dict[str, str]], tuple[Any, ...]],
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

        valset_out = valset_evaluator(seed_candidate)
        outputs, instance_scores, subscores = unpack_evaluation_output(valset_out)
        objective_scores = aggregate_objective_scores(subscores)
        frontier_labels, frontier_scores = compute_frontier_dimensions(
            frontier_type,
            instance_scores,
            objective_scores,
        )

        if run_dir is not None:
            write_eval_output_to_directory(instance_scores, os.path.join(run_dir, "generated_best_outputs_valset"))
        num_evals_run += len(instance_scores)

        gepa_state = GEPAState(
            seed_candidate,
            outputs,
            instance_scores,
            frontier_type=frontier_type,
            frontier_dimension_labels=frontier_labels,
            base_frontier_scores=frontier_scores,
            objective_scores=objective_scores,
            track_best_outputs=track_best_outputs,
        )

        gepa_state.num_full_ds_evals = 1
        gepa_state.total_num_evals = num_evals_run

    return gepa_state
