import os
import random
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from gepa.core.state import GEPAState
from gepa.gepa_utils import scores_sum
from gepa.strategies.candidate_selector import (
    MultiObjectiveCandidateSelector,
    ParetoCandidateSelector,
)


def _make_multi_state():
    seed = {"p": "base"}
    base_scores = [
        {"correct": 0.5, "fun": 0.5},
        {"correct": 0.5, "fun": 0.5},
    ]
    state = GEPAState(seed, ([], base_scores))
    s1 = [
        {"correct": 0.9, "fun": 0.2},
        {"correct": 0.8, "fun": 0.1},
    ]
    val1 = scores_sum(s1) / 2
    state.update_state_with_new_program(
        parent_program_idx=[0],
        new_program={"p": "c1"},
        valset_score=val1,
        valset_outputs=[],
        valset_subscores=s1,
        run_dir=None,
        num_metric_calls_by_discovery_of_new_program=0,
    )
    s2 = [
        {"correct": 0.2, "fun": 0.9},
        {"correct": 0.1, "fun": 0.8},
    ]
    val2 = scores_sum(s2) / 2
    state.update_state_with_new_program(
        parent_program_idx=[0],
        new_program={"p": "c2"},
        valset_score=val2,
        valset_outputs=[],
        valset_subscores=s2,
        run_dir=None,
        num_metric_calls_by_discovery_of_new_program=0,
    )
    return state


def test_hybrid_nondomination():
    state = _make_multi_state()
    selector = MultiObjectiveCandidateSelector(
        rng=random.Random(0),
        objectives=["correct", "fun"],
        selection_strategy="hybrid",
        normalize="zscore",
        global_bonus=3,
        instance_sample_size=None,
    )
    counts = {0: 0, 1: 0, 2: 0}
    for _ in range(200):
        idx = selector.select_candidate_idx(state)
        counts[idx] += 1
    assert counts[1] > 0 and counts[2] > 0
    assert counts[0] < counts[1] and counts[0] < counts[2]


def test_scalar_fallback_matches_pareto():
    seed = {"p": "base"}
    base_scores = [0.1, 0.2]
    state = GEPAState(seed, ([], base_scores))
    s1 = [0.5, 0.3]
    val1 = sum(s1) / 2
    state.update_state_with_new_program(
        parent_program_idx=[0],
        new_program={"p": "c1"},
        valset_score=val1,
        valset_outputs=[],
        valset_subscores=s1,
        run_dir=None,
        num_metric_calls_by_discovery_of_new_program=0,
    )
    rng1 = random.Random(0)
    idx_old = ParetoCandidateSelector(rng1).select_candidate_idx(state)
    rng2 = random.Random(0)
    selector_new = MultiObjectiveCandidateSelector(
        rng=rng2,
        objectives=None,
        selection_strategy="auto",
        normalize="zscore",
        global_bonus=3,
        instance_sample_size=None,
    )
    idx_new = selector_new.select_candidate_idx(state)
    assert idx_new == idx_old


def test_single_objective_dict_equivalence():
    seed = {"p": "base"}
    base_scores = [{"correct": 0.1}, {"correct": 0.2}]
    state = GEPAState(seed, ([], base_scores))
    s1 = [{"correct": 0.5}, {"correct": 0.3}]
    val1 = scores_sum(s1) / 2
    state.update_state_with_new_program(
        parent_program_idx=[0],
        new_program={"p": "c1"},
        valset_score=val1,
        valset_outputs=[],
        valset_subscores=s1,
        run_dir=None,
        num_metric_calls_by_discovery_of_new_program=0,
    )
    rng1 = random.Random(0)
    idx_old = ParetoCandidateSelector(rng1).select_candidate_idx(state)
    rng2 = random.Random(0)
    selector_new = MultiObjectiveCandidateSelector(
        rng=rng2,
        objectives=["correct"],
        selection_strategy="objective_pareto",
        normalize="zscore",
        global_bonus=3,
        instance_sample_size=None,
    )
    idx_new = selector_new.select_candidate_idx(state)
    assert idx_new == idx_old
