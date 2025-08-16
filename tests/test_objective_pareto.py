import random
from types import SimpleNamespace

from gepa.strategies.candidate_selector import ParetoCandidateSelector


class DummyState(SimpleNamespace):
    pass


def make_state(fronts, subscores, is_multi):
    num_cands = len(subscores)
    return DummyState(
        program_at_pareto_front_valset=fronts,
        per_program_tracked_scores=[0.0] * num_cands,
        prog_candidate_val_subscores=subscores,
        program_candidates=[{}] * num_cands,
        is_multi_objective=is_multi,
    )


def test_hybrid_weights_nondomination():
    fronts = [
        {2},
        {0, 1, 2},
    ]
    subscores = [
        [
            {"correct": 1, "fun": 0},
            {"correct": 1, "fun": 0.2},
        ],
        [
            {"correct": 0, "fun": 1},
            {"correct": 0.2, "fun": 1},
        ],
        [
            {"correct": 0.6, "fun": 0.6},
            {"correct": 0.6, "fun": 0.6},
        ],
    ]
    state = make_state(fronts, subscores, True)
    selector = ParetoCandidateSelector(
        rng=random.Random(0),
        selection_strategy="hybrid",
        global_bonus=3,
    )
    weights = selector._compute_weights(state, "hybrid")
    assert weights == {0: 4, 1: 4, 2: 5}


def test_scalar_fallback_auto():
    fronts = [{0, 1}, {1}]
    state = make_state(fronts, [[], []], False)
    selector = ParetoCandidateSelector(rng=random.Random(0), selection_strategy="auto")
    assert selector.select_candidate_idx(state) == 1


def test_single_objective_equivalence():
    fronts = [{0}, {0}]
    subscores = [
        [{"correct": 0.9}, {"correct": 0.8}],
        [{"correct": 0.5}, {"correct": 0.4}],
    ]
    state = make_state(fronts, subscores, True)
    selector = ParetoCandidateSelector(
        rng=random.Random(0), selection_strategy="objective_pareto", objectives=["correct"]
    )
    assert selector.select_candidate_idx(state) == 0
