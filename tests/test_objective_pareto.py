import random
from gepa.strategies.candidate_selector import ParetoCandidateSelector
from gepa.core.state import GEPAState


def make_state_dict(scores_per_candidate):
    seed_candidate = {"c":"x"}
    base_outputs = [None]*len(scores_per_candidate[0])
    state = GEPAState(seed_candidate, (base_outputs, scores_per_candidate[0]))
    for idx, sc in enumerate(scores_per_candidate[1:], start=1):
        obj_scores = None
        if state.is_multi_objective:
            obj_scores = {k: sum(s[k] for s in sc)/len(sc) for k in sc[0].keys()}
            val_score = sum(obj_scores.values())/len(obj_scores)
        else:
            val_score = sum(sc)/len(sc)
        state.update_state_with_new_program(
            parent_program_idx=[0],
            new_program={"c":f"x{idx}"},
            valset_score=val_score,
            valset_outputs=base_outputs,
            valset_subscores=sc,
            run_dir=None,
            num_metric_calls_by_discovery_of_new_program=0,
            valset_objective_scores=obj_scores,
        )
    return state


def test_nondomination_hybrid():
    scores = [
        [
            {"correct":0.1,"fun":0.1},
            {"correct":0.1,"fun":0.1},
        ],
        [
            {"correct":1.0,"fun":0.2},
            {"correct":0.6,"fun":0.1},
        ],
        [
            {"correct":0.2,"fun":0.9},
            {"correct":0.3,"fun":0.8},
        ],
    ]
    state = make_state_dict(scores)
    selector = ParetoCandidateSelector(
        rng=random.Random(0),
        selection_strategy="hybrid",
        global_bonus=3,
    )
    counts = {0:0,1:0,2:0}
    for _ in range(200):
        idx = selector.select_candidate_idx(state)
        counts[idx]+=1
    assert counts[0]==0
    assert counts[1]>0 and counts[2]>0


def test_scalar_fallback_matches_previous():
    scores = [
        [0.5,0.3],
        [0.6,0.2],
        [0.4,0.7],
    ]
    state = make_state_dict(scores)
    rng1 = random.Random(42)
    selector = ParetoCandidateSelector(rng=rng1, selection_strategy="auto")
    chosen = selector.select_candidate_idx(state)

    rng2 = random.Random(42)
    from gepa.gepa_utils import select_program_candidate_from_pareto_front
    expected = select_program_candidate_from_pareto_front(
        state.program_at_pareto_front_valset,
        state.per_program_tracked_scores,
        rng2,
    )
    assert chosen == expected


def test_single_objective_equivalence():
    scores = [
        [
            {"correct":0.5},
            {"correct":0.5},
        ],
        [
            {"correct":0.7},
            {"correct":0.6},
        ],
        [
            {"correct":0.4},
            {"correct":0.8},
        ],
    ]
    state = make_state_dict(scores)
    rng1 = random.Random(1)
    selector = ParetoCandidateSelector(rng=rng1, selection_strategy="objective_pareto")
    chosen = selector.select_candidate_idx(state)

    rng2 = random.Random(1)
    from gepa.gepa_utils import select_program_candidate_from_pareto_front
    expected = select_program_candidate_from_pareto_front(
        state.program_at_pareto_front_valset,
        state.per_program_tracked_scores,
        rng2,
    )
    assert chosen == expected
