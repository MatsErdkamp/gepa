# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa



def json_default(x):
    """Default JSON encoder for objects that are not serializable by default."""
    try:
        return {**x}
    except:
        return repr(x)

def idxmax(lst: list[float]) -> int:
    """Return the index of the maximum value in a list."""
    max_val = max(lst)
    return lst.index(max_val)


def scalarize_score(score):
    """Convert a possibly vector-valued score to a scalar."""
    if isinstance(score, dict):
        if len(score) == 0:
            return 0.0
        return sum(score.values()) / len(score)
    return score


def sum_scores(scores: list) -> float:
    """Sum a list of scores that may be floats or dicts."""
    return sum(scalarize_score(s) for s in scores)


def normalize_objective_scores(
    program_objective_scores: list[dict[str, float]],
    method: str,
    objectives: list[str],
):
    """Normalize per-objective scores across programs."""
    normalized: list[dict[str, float]] = []
    stats: dict[str, tuple[float, float]] = {}
    for obj in objectives:
        vals = [p.get(obj, 0.0) for p in program_objective_scores]
        if method == "minmax":
            min_v, max_v = min(vals), max(vals)
            denom = max_v - min_v if max_v != min_v else 1.0
            stats[obj] = (min_v, denom)
        else:  # zscore
            mean_v = sum(vals) / len(vals)
            var = sum((v - mean_v) ** 2 for v in vals) / len(vals)
            std = var ** 0.5 if var > 0 else 1.0
            stats[obj] = (mean_v, std)
    for p in program_objective_scores:
        norm_p: dict[str, float] = {}
        for obj in objectives:
            offset, denom = stats[obj]
            v = p.get(obj, 0.0)
            if method == "minmax":
                norm_p[obj] = (v - offset) / denom if denom != 0 else 0.0
            else:
                norm_p[obj] = (v - offset) / denom if denom != 0 else 0.0
        normalized.append(norm_p)
    return normalized


def get_objective_front(
    program_objective_scores: list[dict[str, float]],
    method: str,
    objectives: list[str],
):
    """Return set of program indices on the objective-based Pareto front."""
    normalized = normalize_objective_scores(program_objective_scores, method, objectives)
    nondominated: set[int] = set()
    for i, p in enumerate(normalized):
        dominated = False
        for j, q in enumerate(normalized):
            if i == j:
                continue
            if all(q[o] >= p[o] for o in objectives) and any(q[o] > p[o] for o in objectives):
                dominated = True
                break
        if not dominated:
            nondominated.add(i)
    return nondominated


def select_program_candidate_from_objective_pareto_front(
    program_objective_scores: list[dict[str, float]],
    rng,
    method: str,
    objectives: list[str],
):
    """Select a program from the objective-based Pareto front."""
    front = list(get_objective_front(program_objective_scores, method, objectives))
    assert len(front) > 0
    return rng.choice(front)


def select_program_candidate_from_hybrid_front(
    instance_fronts: list[set[int]],
    program_objective_scores: list[dict[str, float]],
    rng,
    method: str,
    objectives: list[str],
    global_bonus: int,
    instance_sample_size: int | None = None,
):
    """Select a program balancing instance- and objective-level diversity."""
    if instance_sample_size is not None:
        sampled_fronts = rng.sample(instance_fronts, k=min(len(instance_fronts), instance_sample_size))
    else:
        sampled_fronts = instance_fronts
    freq: dict[int, int] = {}
    for front in sampled_fronts:
        for idx in front:
            freq[idx] = freq.get(idx, 0) + 1
    objective_front = get_objective_front(program_objective_scores, method, objectives)
    for idx in objective_front:
        freq[idx] = freq.get(idx, 0) + global_bonus
    sampling_list = [idx for idx, f in freq.items() for _ in range(f)]
    assert len(sampling_list) > 0
    return rng.choice(sampling_list)

def is_dominated(y, programs, program_at_pareto_front_valset):
    y_fronts = [front for front in program_at_pareto_front_valset if y in front]
    for front in y_fronts:
        found_dominator_in_front = False
        for other_prog in front:
            if other_prog in programs:
                found_dominator_in_front = True
                break
        if not found_dominator_in_front:
            return False

    return True

def remove_dominated_programs(program_at_pareto_front_valset, scores=None):
    freq = {}
    for front in program_at_pareto_front_valset:
        for p in front:
            freq[p] = freq.get(p, 0) + 1

    dominated = set()
    programs = list(freq.keys())

    if scores is None:
        scores = dict.fromkeys(programs, 1)

    programs = sorted(programs, key=lambda x: scores[x], reverse=False)

    found_to_remove = True
    while found_to_remove:
        found_to_remove = False
        for y in programs:
            if y in dominated:
                continue
            if is_dominated(y, set(programs).difference({y}).difference(dominated), program_at_pareto_front_valset):
                dominated.add(y)
                found_to_remove = True
                break

    dominators = [p for p in programs if p not in dominated]
    for front in program_at_pareto_front_valset:
        assert any(p in front for p in dominators)

    new_program_at_pareto_front_valset = [{prog_idx for prog_idx in front if prog_idx in dominators} for front in program_at_pareto_front_valset]
    assert len(new_program_at_pareto_front_valset) == len(program_at_pareto_front_valset)
    for front_old, front_new in zip(program_at_pareto_front_valset, new_program_at_pareto_front_valset, strict=False):
        assert front_new.issubset(front_old)

    return new_program_at_pareto_front_valset

def find_dominator_programs(pareto_front_programs, train_val_weighted_agg_scores_for_all_programs):
    train_val_pareto_front_programs = pareto_front_programs
    new_program_at_pareto_front_valset = remove_dominated_programs(train_val_pareto_front_programs, scores=train_val_weighted_agg_scores_for_all_programs)
    uniq_progs = []
    for front in new_program_at_pareto_front_valset:
        uniq_progs.extend(front)
    uniq_progs = set(uniq_progs)
    return list(uniq_progs)

def select_program_candidate_from_pareto_front(pareto_front_programs, train_val_weighted_agg_scores_for_all_programs, rng):
    train_val_pareto_front_programs = pareto_front_programs
    new_program_at_pareto_front_valset = remove_dominated_programs(train_val_pareto_front_programs, scores=train_val_weighted_agg_scores_for_all_programs)
    program_frequency_in_validation_pareto_front = {}
    for testcase_pareto_front in new_program_at_pareto_front_valset:
        for prog_idx in testcase_pareto_front:
            if prog_idx not in program_frequency_in_validation_pareto_front:
                program_frequency_in_validation_pareto_front[prog_idx] = 0
            program_frequency_in_validation_pareto_front[prog_idx] += 1

    sampling_list = [prog_idx for prog_idx, freq in program_frequency_in_validation_pareto_front.items() for _ in range(freq)]
    assert len(sampling_list) > 0
    curr_prog_id = rng.choice(sampling_list)
    return curr_prog_id
