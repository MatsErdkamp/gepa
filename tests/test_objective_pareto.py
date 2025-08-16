import random
from types import SimpleNamespace

from gepa.core.state import GEPAState
import random
from types import SimpleNamespace

from gepa.core.state import GEPAState
from gepa.strategies.candidate_selector import ParetoCandidateSelector
from gepa.gepa_utils import select_program_candidate_from_pareto_front


def _build_hybrid_state():
    seed = {}
    outputs = [None, None]
    base_scores = [
        {"correct": 1.0, "fun": 0.0},
        {"correct": 0.2, "fun": 0.3},
    ]
    state = GEPAState(seed, (outputs, base_scores), objectives=["correct", "fun"])

    cand1_scores = [
        {"correct": 0.0, "fun": 0.2},
        {"correct": 0.0, "fun": 1.0},
    ]
    val1 = (0.1 + 0.5) / 2
    state.update_state_with_new_program([0], {}, val1, [None, None], cand1_scores, None, 0)

    cand2_scores = [
        {"correct": 0.4, "fun": 0.4},
        {"correct": 0.4, "fun": 0.4},
    ]
    val2 = (0.4 + 0.4) / 2
    state.update_state_with_new_program([0], {}, val2, [None, None], cand2_scores, None, 0)
    return state


def test_hybrid_sampling_preserves_specialists():
    state = _build_hybrid_state()
    selector = ParetoCandidateSelector(
        rng=random.Random(0),
        objectives=["correct", "fun"],
        selection_strategy="hybrid",
        global_bonus=3,
    )
    counts = {0: 0, 1: 0, 2: 0}
    for _ in range(900):
        idx = selector.select_candidate_idx(state)
        counts[idx] += 1
    assert counts[2] > 0
    assert counts[0] > counts[2]
    assert counts[1] > counts[2]


def test_scalar_fallback_matches_old_sampling():
    fronts = [{0, 1}, {1, 2}]
    scores = [0.5, 0.8, 0.3]
    state = SimpleNamespace(
        program_at_pareto_front_valset=fronts,
        per_program_tracked_scores=scores,
        program_candidates=[None, None, None],
        is_multi_objective=False,
        objectives=None,
    )
    selector = ParetoCandidateSelector(rng=random.Random(0), selection_strategy="auto")
    expected = select_program_candidate_from_pareto_front(fronts, scores, random.Random(0))
    assert selector.select_candidate_idx(state) == expected


def test_single_objective_equivalence():
    fronts = [{0, 1}, {1, 2}]
    scores = [0.5, 0.8, 0.3]
    state = SimpleNamespace(
        program_at_pareto_front_valset=fronts,
        per_program_tracked_scores=scores,
        program_candidates=[None, None, None],
        is_multi_objective=False,
        objectives=["correct"],
    )
    selector = ParetoCandidateSelector(
        rng=random.Random(0),
        selection_strategy="objective_pareto",
        objectives=["correct"],
    )
    expected = select_program_candidate_from_pareto_front(fronts, scores, random.Random(0))
    assert selector.select_candidate_idx(state) == expected
