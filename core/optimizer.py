"""
core.optimizer
---------------
A compact, mostly dependency-free (numpy + scipy/sklearn already required by
the app) optimization library used to search the surrogate model(s) trained
in Step 2 for a Pareto-optimal set of designs.

Algorithms, organized the way the literature usually classifies them:

METAHEURISTIC (population-based direct search over the surrogate)
    - run_nsga2                     NSGA-II
    - run_nsga3                     NSGA-III (reference-point, many-objective)
    - run_steady_state_nsga2        Steady-State NSGA-II ("S-NSGA-II")
    - run_hype                      HypE (hypervolume-indicator many-objective EA)
    - run_genetic_algorithm         Real-coded GA (weighted-sum scalarization)
    - run_aco                       Ant Colony Optimization for continuous domains (ACOR)
    - run_random_search             Monte Carlo baseline

MODEL-BASED / SURROGATE-ASSISTED (a *second*, cheap meta-surrogate is fit on
the fly to reduce how many times the real objective is called; this is the
standard "surrogate-based optimization" / SBO loop)
    - run_rbf_surrogate_assisted     RBFMOpt-style (RBF interpolant + inner NSGA-II)
    - run_ann_surrogate_assisted     ANN-assisted (MLP interpolant + inner NSGA-II)
    - run_differential_evolution     DE + weighted scalarization (smooth continuous problems)

All of the above share the same calling convention:
    result = run_xxx(objective_fn, bounds, directions, ..., seed=42, progress_callback=fn)
    result.X -> (n, d) decision variables on the returned Pareto front
    result.F -> (n, k) objective values (natural units) on that front

Reproducibility note: every function below takes an explicit `seed` and
routes ALL of its randomness through a *local* `numpy.random.Generator`
instance created from that seed (`np.random.default_rng(seed)`), or through
an explicitly-seeded call into scipy/sklearn. None of these functions read or
write NumPy's legacy global random state, so two calls with the same seed
always produce the same result, regardless of what else has happened earlier
in the same Python/Streamlit process.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb, pi, sqrt
from typing import Callable, Optional, Sequence

import numpy as np


@dataclass
class OptimizationResult:
    X: np.ndarray  # decision variables of the final Pareto front, shape (n, d)
    F: np.ndarray  # objective values of the final Pareto front (already sign-corrected), shape (n, k)


# ---------------------------------------------------------------------------
# Shared building blocks (non-dominated sorting, crowding, EA operators)
# ---------------------------------------------------------------------------

def _fast_non_dominated_sort(F: np.ndarray):
    n = F.shape[0]
    dominates = [[] for _ in range(n)]
    dominated_count = np.zeros(n, dtype=int)
    fronts = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(F[p], F[q]):
                dominates[p].append(q)
            elif _dominates(F[q], F[p]):
                dominated_count[p] += 1
        if dominated_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominates[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    fronts.pop()  # last one is always empty
    return fronts


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def _crowding_distance(F_front: np.ndarray) -> np.ndarray:
    n, k = F_front.shape
    distance = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)

    for m in range(k):
        order = np.argsort(F_front[:, m])
        distance[order[0]] = np.inf
        distance[order[-1]] = np.inf
        f_min, f_max = F_front[order[0], m], F_front[order[-1], m]
        span = f_max - f_min
        if span <= 1e-12:
            continue
        for idx in range(1, n - 1):
            prev_val = F_front[order[idx - 1], m]
            next_val = F_front[order[idx + 1], m]
            distance[order[idx]] += (next_val - prev_val) / span
    return distance


def _tournament_select(pop_rank, pop_secondary, rng: np.random.Generator, k: int = 2) -> int:
    """Binary tournament: lower rank wins; ties broken by HIGHER secondary
    score (crowding distance, or a hypervolume-contribution proxy)."""
    n = len(pop_rank)
    candidates = rng.integers(0, n, size=k)
    best = candidates[0]
    for c in candidates[1:]:
        if pop_rank[c] < pop_rank[best]:
            best = c
        elif pop_rank[c] == pop_rank[best] and pop_secondary[c] > pop_secondary[best]:
            best = c
    return int(best)


def _sbx_crossover(p1, p2, lb, ub, rng: np.random.Generator, eta: float = 15.0, prob: float = 0.9):
    if rng.random() > prob:
        return p1.copy(), p2.copy()
    c1, c2 = p1.copy(), p2.copy()
    for i in range(len(p1)):
        if rng.random() > 0.5 or abs(p1[i] - p2[i]) < 1e-14:
            continue
        x1, x2 = min(p1[i], p2[i]), max(p1[i], p2[i])
        rand = rng.random()
        beta = 1.0 + (2.0 * (x1 - lb[i]) / max(x2 - x1, 1e-14))
        alpha = 2.0 - beta ** (-(eta + 1.0))
        if rand <= 1.0 / alpha:
            beta_q = (rand * alpha) ** (1.0 / (eta + 1.0))
        else:
            beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
        child1 = 0.5 * ((x1 + x2) - beta_q * (x2 - x1))
        child2 = 0.5 * ((x1 + x2) + beta_q * (x2 - x1))
        c1[i] = np.clip(child1, lb[i], ub[i])
        c2[i] = np.clip(child2, lb[i], ub[i])
    return c1, c2


def _poly_mutation(x, lb, ub, rng: np.random.Generator, eta: float = 20.0, prob: Optional[float] = None):
    d = len(x)
    if prob is None:
        prob = 1.0 / d
    y = x.copy()
    for i in range(d):
        if rng.random() > prob or ub[i] - lb[i] < 1e-14:
            continue
        delta1 = (y[i] - lb[i]) / (ub[i] - lb[i])
        delta2 = (ub[i] - y[i]) / (ub[i] - lb[i])
        rand = rng.random()
        mut_pow = 1.0 / (eta + 1.0)
        if rand < 0.5:
            xy = 1.0 - delta1
            val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
            delta_q = val ** mut_pow - 1.0
        else:
            xy = 1.0 - delta2
            val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
            delta_q = 1.0 - val ** mut_pow
        y[i] = np.clip(y[i] + delta_q * (ub[i] - lb[i]), lb[i], ub[i])
    return y


def _generate_offspring(pop, rank, secondary, lb, ub, rng: np.random.Generator, n_offspring: int):
    offspring = []
    while len(offspring) < n_offspring:
        i1 = _tournament_select(rank, secondary, rng)
        i2 = _tournament_select(rank, secondary, rng)
        c1, c2 = _sbx_crossover(pop[i1], pop[i2], lb, ub, rng)
        offspring.append(_poly_mutation(c1, lb, ub, rng))
        if len(offspring) < n_offspring:
            offspring.append(_poly_mutation(c2, lb, ub, rng))
    return np.array(offspring)


def _environmental_selection_crowding(combined_X, combined_F, pop_size):
    """Standard NSGA-II survivor selection: fill whole fronts, break the
    boundary front by crowding distance (higher = more space around it)."""
    fronts = _fast_non_dominated_sort(combined_F)
    new_indices = []
    for front in fronts:
        if len(new_indices) + len(front) <= pop_size:
            new_indices.extend(front)
        else:
            remaining = pop_size - len(new_indices)
            cd = _crowding_distance(combined_F[front])
            order = np.argsort(-cd)
            new_indices.extend([front[k] for k in order[:remaining]])
            break
    return combined_X[new_indices], combined_F[new_indices]


def pareto_front(F: np.ndarray) -> np.ndarray:
    """Return the indices of the non-dominated (Pareto-optimal) rows of F.
    F is assumed to already be in MINIMIZE-everything convention.
    """
    if len(F) == 0:
        return np.array([], dtype=int)
    fronts = _fast_non_dominated_sort(F)
    return np.array(fronts[0], dtype=int)


def _normalize_probe(objective_fn, lb, ub, sign, rng: np.random.Generator, n_probe: int):
    """Cheap random probe used to rescale objectives onto a comparable ~[0, 1]
    range before combining them with weights (weighted-sum scalarization is
    meaningless if objectives have wildly different natural units/scales).
    """
    probe_x = lb + rng.random((n_probe, len(lb))) * (ub - lb)
    probe_f = np.atleast_2d(np.asarray(objective_fn(probe_x), dtype=float)) * sign
    obj_lo = probe_f.min(axis=0)
    obj_span = np.clip(probe_f.max(axis=0) - obj_lo, 1e-9, None)
    return obj_lo, obj_span


def _make_weight_sets(n_obj: int, n_weight_sets: int, rng: np.random.Generator) -> np.ndarray:
    if n_obj == 1:
        return np.array([[1.0]])
    if n_obj == 2:
        w = np.linspace(0.0, 1.0, max(2, n_weight_sets))
        return np.stack([w, 1.0 - w], axis=1)
    return rng.dirichlet(np.ones(n_obj), size=max(2, n_weight_sets))


# ---------------------------------------------------------------------------
# METAHEURISTIC: NSGA-II
# ---------------------------------------------------------------------------

def run_nsga2(
    objective_fn: Callable[[np.ndarray], np.ndarray],
    bounds: Sequence[tuple],
    directions: Sequence[str],
    pop_size: int = 100,
    n_gen: int = 50,
    seed: int = 42,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> OptimizationResult:
    """NSGA-II (Deb et al., 2002). General-purpose, the de-facto default for
    multi-objective problems (2-3 objectives), robust on rugged/discontinuous
    surrogates (tree ensembles), moderate computational cost.
    """
    rng = np.random.default_rng(seed)
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    d = len(bounds)
    sign = np.array([1.0 if dirn == "min" else -1.0 for dirn in directions])

    def evaluate(X):
        raw = np.atleast_2d(np.asarray(objective_fn(X), dtype=float))
        return raw * sign

    pop = lb + rng.random((pop_size, d)) * (ub - lb)
    F = evaluate(pop)

    for gen in range(n_gen):
        fronts = _fast_non_dominated_sort(F)
        rank = np.zeros(len(pop), dtype=int)
        crowd = np.zeros(len(pop))
        for r, front in enumerate(fronts):
            rank[front] = r
            crowd[front] = _crowding_distance(F[front])

        offspring = _generate_offspring(pop, rank, crowd, lb, ub, rng, pop_size)
        F_off = evaluate(offspring)

        combined_X = np.vstack([pop, offspring])
        combined_F = np.vstack([F, F_off])
        pop, F = _environmental_selection_crowding(combined_X, combined_F, pop_size)

        if progress_callback is not None:
            progress_callback(gen + 1, n_gen)

    fronts = _fast_non_dominated_sort(F)
    best_front = fronts[0]
    return OptimizationResult(X=pop[best_front], F=F[best_front] * sign)


# ---------------------------------------------------------------------------
# METAHEURISTIC: NSGA-III (reference-point, many-objective)
# ---------------------------------------------------------------------------

def _das_dennis_points(n_obj: int, divisions: int) -> np.ndarray:
    def _gen(n, total):
        if n == 1:
            yield (total,)
            return
        for i in range(total + 1):
            for tail in _gen(n - 1, total - i):
                yield (i,) + tail

    pts = np.array(list(_gen(n_obj, divisions)), dtype=float)
    return pts / divisions


def _reference_points(n_obj: int, target_points: int = 100) -> np.ndarray:
    if n_obj == 2:
        w = np.linspace(0.0, 1.0, max(2, target_points))
        return np.stack([w, 1.0 - w], axis=1)
    divisions = 1
    while divisions < 20 and comb(divisions + n_obj - 1, n_obj - 1) < min(target_points, 300):
        divisions += 1
    return _das_dennis_points(n_obj, divisions)


def _associate_to_reference_points(F_norm: np.ndarray, ref_points: np.ndarray):
    """Perpendicular distance of each (normalized) objective vector to each
    reference direction; returns (closest_ref_idx, distance) per point."""
    ref_norm = ref_points / np.clip(np.linalg.norm(ref_points, axis=1, keepdims=True), 1e-12, None)
    proj_len = F_norm @ ref_norm.T  # (n, n_ref)
    proj = proj_len[:, :, None] * ref_norm[None, :, :]  # (n, n_ref, k)
    perp = F_norm[:, None, :] - proj
    dist = np.linalg.norm(perp, axis=2)  # (n, n_ref)
    closest = np.argmin(dist, axis=1)
    return closest, dist[np.arange(len(F_norm)), closest]


def _environmental_selection_nsga3(combined_X, combined_F, pop_size, ref_points):
    fronts = _fast_non_dominated_sort(combined_F)
    accepted_indices = []
    boundary_front = []
    for front in fronts:
        if len(accepted_indices) + len(front) <= pop_size:
            accepted_indices.extend(front)
        else:
            boundary_front = front
            break

    if not boundary_front:
        return combined_X[accepted_indices], combined_F[accepted_indices]

    all_considered = accepted_indices + boundary_front
    F_considered = combined_F[all_considered]
    f_min = F_considered.min(axis=0)
    f_max = F_considered.max(axis=0)
    span = np.clip(f_max - f_min, 1e-12, None)
    F_norm = (F_considered - f_min) / span

    closest, _ = _associate_to_reference_points(F_norm, ref_points)
    n_accepted = len(accepted_indices)
    niche_of_accepted = closest[:n_accepted]
    niche_counts = np.bincount(niche_of_accepted, minlength=len(ref_points)).astype(np.int64)

    remaining_slots = pop_size - n_accepted
    boundary_local_idx = list(range(n_accepted, len(all_considered)))
    boundary_niche = closest[n_accepted:]
    selected_local = []
    available = set(boundary_local_idx)

    while remaining_slots > 0 and available:
        min_count = niche_counts.min()
        candidate_niches = np.where(niche_counts == min_count)[0]
        chosen_niche = candidate_niches[0]
        members = [i for i in available if boundary_niche[boundary_local_idx.index(i)] == chosen_niche]
        if not members:
            niche_counts[chosen_niche] = np.iinfo(np.int64).max  # exhausted niche, never pick again
            continue
        pick = members[0]
        selected_local.append(pick)
        available.remove(pick)
        niche_counts[chosen_niche] += 1
        remaining_slots -= 1

    final_indices = accepted_indices + [all_considered[i] for i in selected_local]
    return combined_X[final_indices], combined_F[final_indices]


def run_nsga3(
    objective_fn: Callable[[np.ndarray], np.ndarray],
    bounds: Sequence[tuple],
    directions: Sequence[str],
    pop_size: int = 100,
    n_gen: int = 50,
    seed: int = 42,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> OptimizationResult:
    """NSGA-III (Deb & Jain, 2014). Reference-point-based survivor selection
    instead of crowding distance — designed for MANY-objective problems
    (4+ targets), where plain NSGA-II's crowding distance loses resolution.
    Normalization here is simplified (min-max instead of the full
    hyperplane-intercept method from the original paper); still effective in
    practice, but slightly less precise on extreme, badly-scaled fronts.
    """
    rng = np.random.default_rng(seed)
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    d = len(bounds)
    n_obj = len(directions)
    sign = np.array([1.0 if dirn == "min" else -1.0 for dirn in directions])
    ref_points = _reference_points(n_obj, target_points=pop_size)

    def evaluate(X):
        raw = np.atleast_2d(np.asarray(objective_fn(X), dtype=float))
        return raw * sign

    pop = lb + rng.random((pop_size, d)) * (ub - lb)
    F = evaluate(pop)

    for gen in range(n_gen):
        fronts = _fast_non_dominated_sort(F)
        rank = np.zeros(len(pop), dtype=int)
        for r, front in enumerate(fronts):
            rank[front] = r
        crowd = np.zeros(len(pop))
        for front in fronts:
            crowd[front] = _crowding_distance(F[front])

        offspring = _generate_offspring(pop, rank, crowd, lb, ub, rng, pop_size)
        F_off = evaluate(offspring)

        combined_X = np.vstack([pop, offspring])
        combined_F = np.vstack([F, F_off])
        pop, F = _environmental_selection_nsga3(combined_X, combined_F, pop_size, ref_points)

        if progress_callback is not None:
            progress_callback(gen + 1, n_gen)

    front_idx = pareto_front(F)
    return OptimizationResult(X=pop[front_idx], F=F[front_idx] * sign)


# ---------------------------------------------------------------------------
# METAHEURISTIC: Steady-State NSGA-II ("S-NSGA-II")
# ---------------------------------------------------------------------------

def run_steady_state_nsga2(
    objective_fn: Callable[[np.ndarray], np.ndarray],
    bounds: Sequence[tuple],
    directions: Sequence[str],
    pop_size: int = 80,
    n_iterations: int = 1500,
    seed: int = 42,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> OptimizationResult:
    """Steady-State NSGA-II: replaces ONE individual at a time (rather than a
    whole generation at once) via the same rank+crowding survivor selection.
    Tends to converge in fewer total objective evaluations than generational
    NSGA-II — useful when each evaluation is relatively costly — at the cost
    of more (cheaper) bookkeeping steps and slightly higher variance run to
    run for a fixed evaluation budget.
    """
    rng = np.random.default_rng(seed)
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    d = len(bounds)
    sign = np.array([1.0 if dirn == "min" else -1.0 for dirn in directions])

    def evaluate(X):
        raw = np.atleast_2d(np.asarray(objective_fn(X), dtype=float))
        return raw * sign

    pop = lb + rng.random((pop_size, d)) * (ub - lb)
    F = evaluate(pop)

    for it in range(n_iterations):
        fronts = _fast_non_dominated_sort(F)
        rank = np.zeros(len(pop), dtype=int)
        crowd = np.zeros(len(pop))
        for r, front in enumerate(fronts):
            rank[front] = r
            crowd[front] = _crowding_distance(F[front])

        i1 = _tournament_select(rank, crowd, rng)
        i2 = _tournament_select(rank, crowd, rng)
        c1, _ = _sbx_crossover(pop[i1], pop[i2], lb, ub, rng)
        child = _poly_mutation(c1, lb, ub, rng)
        f_child = evaluate(child.reshape(1, -1))

        combined_X = np.vstack([pop, child.reshape(1, -1)])
        combined_F = np.vstack([F, f_child])
        pop, F = _environmental_selection_crowding(combined_X, combined_F, pop_size)

        if progress_callback is not None and (it % max(1, n_iterations // 100) == 0 or it == n_iterations - 1):
            progress_callback(it + 1, n_iterations)

    front_idx = pareto_front(F)
    return OptimizationResult(X=pop[front_idx], F=F[front_idx] * sign)


# ---------------------------------------------------------------------------
# METAHEURISTIC: HypE (hypervolume-indicator many-objective EA)
# ---------------------------------------------------------------------------

def _hv_contribution_montecarlo(F_front: np.ndarray, ref_point: np.ndarray, rng: np.random.Generator, n_samples: int = 500) -> np.ndarray:
    """Monte-Carlo estimate of each point's EXCLUSIVE hypervolume contribution
    within the box [min(F_front), ref_point] (Bader & Zitzler, 2011)."""
    n, k = F_front.shape
    box_lo = F_front.min(axis=0)
    box_hi = ref_point
    span = np.clip(box_hi - box_lo, 1e-12, None)
    samples = box_lo + rng.random((n_samples, k)) * span

    dominated_by = np.zeros((n_samples, n), dtype=bool)
    for i in range(n):
        dominated_by[:, i] = np.all(F_front[i] <= samples, axis=1)
    counts = dominated_by.sum(axis=1)
    covered = counts > 0

    contributions = np.zeros(n)
    for i in range(n):
        mask = dominated_by[:, i] & covered
        if mask.any():
            contributions[i] = np.mean(1.0 / counts[mask])
    volume = float(np.prod(span))
    return contributions * volume


def run_hype(
    objective_fn: Callable[[np.ndarray], np.ndarray],
    bounds: Sequence[tuple],
    directions: Sequence[str],
    pop_size: int = 80,
    n_gen: int = 40,
    hv_samples: int = 400,
    seed: int = 42,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> OptimizationResult:
    """HypE (Bader & Zitzler, 2011). Uses a Monte-Carlo-estimated hypervolume
    contribution instead of crowding distance for both mating and survivor
    selection. Designed for MANY-objective problems where hypervolume is a
    more discriminating quality indicator than crowding distance; more
    computationally expensive per generation than NSGA-II because of the
    Monte-Carlo sampling, so it suits smaller population/generation budgets.
    """
    rng = np.random.default_rng(seed)
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    d = len(bounds)
    sign = np.array([1.0 if dirn == "min" else -1.0 for dirn in directions])

    def evaluate(X):
        raw = np.atleast_2d(np.asarray(objective_fn(X), dtype=float))
        return raw * sign

    def hv_fitness(F_arr, fronts):
        fitness = np.zeros(len(F_arr))
        for front in fronts:
            if len(front) == 0:
                continue
            sub_F = F_arr[front]
            ref = sub_F.max(axis=0) * 1.1 + 1e-6
            fitness[front] = _hv_contribution_montecarlo(sub_F, ref, rng, n_samples=hv_samples)
        return fitness

    pop = lb + rng.random((pop_size, d)) * (ub - lb)
    F = evaluate(pop)

    for gen in range(n_gen):
        fronts = _fast_non_dominated_sort(F)
        rank = np.zeros(len(pop), dtype=int)
        for r, front in enumerate(fronts):
            rank[front] = r
        hv_fit = hv_fitness(F, fronts)

        offspring = _generate_offspring(pop, rank, hv_fit, lb, ub, rng, pop_size)
        F_off = evaluate(offspring)

        combined_X = np.vstack([pop, offspring])
        combined_F = np.vstack([F, F_off])
        combined_fronts = _fast_non_dominated_sort(combined_F)

        new_indices = []
        for front in combined_fronts:
            if len(new_indices) + len(front) <= pop_size:
                new_indices.extend(front)
            else:
                remaining = pop_size - len(new_indices)
                sub_F = combined_F[front]
                ref = sub_F.max(axis=0) * 1.1 + 1e-6
                contrib = _hv_contribution_montecarlo(sub_F, ref, rng, n_samples=hv_samples)
                order = np.argsort(-contrib)
                new_indices.extend([front[k] for k in order[:remaining]])
                break

        pop, F = combined_X[new_indices], combined_F[new_indices]

        if progress_callback is not None:
            progress_callback(gen + 1, n_gen)

    front_idx = pareto_front(F)
    return OptimizationResult(X=pop[front_idx], F=F[front_idx] * sign)


# ---------------------------------------------------------------------------
# Generic wrapper: turn a batched single-objective solver into a
# multi-objective one via weighted-sum scalarization across many weight sets.
# Shared by GA and ACOR.
# ---------------------------------------------------------------------------

def _weighted_scan(
    objective_fn,
    bounds,
    directions,
    inner_solver: Callable[[Callable[[np.ndarray], np.ndarray], np.ndarray, np.ndarray, np.random.Generator], np.ndarray],
    n_weight_sets: int,
    seed: int,
    progress_callback: Optional[Callable[[int, int], None]],
) -> OptimizationResult:
    rng = np.random.default_rng(seed)
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    sign = np.array([1.0 if dirn == "min" else -1.0 for dirn in directions])
    n_obj = len(directions)

    weight_sets = _make_weight_sets(n_obj, n_weight_sets, rng)
    obj_lo, obj_span = _normalize_probe(objective_fn, lb, ub, sign, rng, n_probe=min(60, max(20, 4 * len(bounds))))

    all_X, all_F = [], []
    total = len(weight_sets)
    for idx, weights in enumerate(weight_sets):
        def scalarized_batch(X, weights=weights):
            raw = np.atleast_2d(np.asarray(objective_fn(X), dtype=float))
            normalized = (raw * sign - obj_lo) / obj_span
            return normalized @ weights

        sub_seed = int(rng.integers(0, 1_000_000))
        sub_rng = np.random.default_rng(sub_seed)
        x_best = inner_solver(scalarized_batch, lb, ub, sub_rng)
        f_best = np.asarray(objective_fn(np.atleast_2d(x_best)), dtype=float)[0]
        all_X.append(x_best)
        all_F.append(f_best)
        if progress_callback is not None:
            progress_callback(idx + 1, total)

    X = np.array(all_X)
    F_natural = np.array(all_F)
    F_min = F_natural * sign
    front_idx = pareto_front(F_min)
    return OptimizationResult(X=X[front_idx], F=F_natural[front_idx])


# ---------------------------------------------------------------------------
# METAHEURISTIC: Genetic Algorithm (real-coded, weighted-sum scalarization)
# ---------------------------------------------------------------------------

def run_genetic_algorithm(
    objective_fn: Callable[[np.ndarray], np.ndarray],
    bounds: Sequence[tuple],
    directions: Sequence[str],
    n_weight_sets: int = 12,
    pop_size: int = 60,
    n_gen: int = 80,
    seed: int = 42,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> OptimizationResult:
    """A plain real-coded Genetic Algorithm (SBX crossover + polynomial
    mutation + elitist truncation selection on scalar fitness) — the
    textbook single-objective baseline, extended to multiple objectives via
    weighted-sum scalarization across a spread of weight vectors (like DE).
    Simple, fast, and a useful sanity-check baseline against the
    Pareto-based methods (NSGA-II/III), but weighted-sum scalarization
    cannot find points in non-convex regions of the true Pareto front.
    """

    def inner_ga(scalarized_batch, lb, ub, rng):
        d = len(lb)
        pop = lb + rng.random((pop_size, d)) * (ub - lb)
        fitness = scalarized_batch(pop)
        for _ in range(n_gen):
            rank = np.argsort(fitness)  # rank 0 = best (lowest fitness)
            pseudo_rank = np.empty(pop_size, dtype=int)
            pseudo_rank[rank] = np.arange(pop_size)
            offspring = _generate_offspring(pop, pseudo_rank, -fitness, lb, ub, rng, pop_size)
            off_fitness = scalarized_batch(offspring)
            combined_X = np.vstack([pop, offspring])
            combined_fitness = np.concatenate([fitness, off_fitness])
            order = np.argsort(combined_fitness)[:pop_size]
            pop, fitness = combined_X[order], combined_fitness[order]
        return pop[np.argmin(fitness)]

    return _weighted_scan(objective_fn, bounds, directions, inner_ga, n_weight_sets, seed, progress_callback)


# ---------------------------------------------------------------------------
# METAHEURISTIC: Ant Colony Optimization for continuous domains (ACOR)
# ---------------------------------------------------------------------------

def run_aco(
    objective_fn: Callable[[np.ndarray], np.ndarray],
    bounds: Sequence[tuple],
    directions: Sequence[str],
    n_weight_sets: int = 12,
    archive_size: int = 30,
    n_ants: int = 10,
    n_iterations: int = 60,
    q: float = 0.5,
    xi: float = 0.85,
    seed: int = 42,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> OptimizationResult:
    """Ant Colony Optimization for continuous domains (ACOR; Socha & Dorigo,
    2008). Maintains an archive of solutions and samples new candidates from
    Gaussian kernels centered on archive members, weighted by rank (a
    pheromone-like mechanism carried over from discrete ACO). Extended to
    multiple objectives via weighted-sum scalarization, like GA/DE. A solid,
    literature-standard metaheuristic alternative outside the GA/DE family;
    tends to do well on smooth landscapes with a modest evaluation budget.
    """

    def inner_acor(scalarized_batch, lb, ub, rng):
        d = len(lb)
        archive_X = lb + rng.random((archive_size, d)) * (ub - lb)
        archive_F = scalarized_batch(archive_X)
        order = np.argsort(archive_F)
        archive_X, archive_F = archive_X[order], archive_F[order]

        ranks = np.arange(1, archive_size + 1)
        weights = (1.0 / (q * archive_size * sqrt(2 * pi))) * np.exp(-((ranks - 1) ** 2) / (2 * (q * archive_size) ** 2))
        probs = weights / weights.sum()

        for _ in range(n_iterations):
            new_X = np.zeros((n_ants, d))
            for a in range(n_ants):
                for dim in range(d):
                    j = rng.choice(archive_size, p=probs)
                    mean = archive_X[j, dim]
                    sigma = xi * np.mean(np.abs(archive_X[:, dim] - archive_X[j, dim])) / max(archive_size - 1, 1)
                    sigma = max(sigma, 1e-6 * (ub[dim] - lb[dim]))
                    new_X[a, dim] = np.clip(rng.normal(mean, sigma), lb[dim], ub[dim])
            new_F = scalarized_batch(new_X)
            combined_X = np.vstack([archive_X, new_X])
            combined_F = np.concatenate([archive_F, new_F])
            order = np.argsort(combined_F)[:archive_size]
            archive_X, archive_F = combined_X[order], combined_F[order]
        return archive_X[0]

    return _weighted_scan(objective_fn, bounds, directions, inner_acor, n_weight_sets, seed, progress_callback)


# ---------------------------------------------------------------------------
# METAHEURISTIC: Monte Carlo Random Search (baseline)
# ---------------------------------------------------------------------------

def run_random_search(
    objective_fn: Callable[[np.ndarray], np.ndarray],
    bounds: Sequence[tuple],
    directions: Sequence[str],
    n_samples: int = 3000,
    batch_size: int = 200,
    seed: int = 42,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> OptimizationResult:
    """Monte-Carlo random search: sample the design space uniformly and keep
    the non-dominated designs. No tuning required — a fast, simple baseline
    to sanity-check what the genetic/DE/ACO solvers find.
    """
    rng = np.random.default_rng(seed)
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    d = len(bounds)
    sign = np.array([1.0 if dirn == "min" else -1.0 for dirn in directions])

    X_all = lb + rng.random((n_samples, d)) * (ub - lb)
    F_chunks = []
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        chunk = np.atleast_2d(np.asarray(objective_fn(X_all[start:end]), dtype=float))
        F_chunks.append(chunk)
        if progress_callback is not None:
            progress_callback(end, n_samples)

    F_natural = np.vstack(F_chunks)
    F_min = F_natural * sign
    front_idx = pareto_front(F_min)
    return OptimizationResult(X=X_all[front_idx], F=F_natural[front_idx])


# ---------------------------------------------------------------------------
# MODEL-BASED / HYBRID: Differential Evolution + weighted scalarization
# ---------------------------------------------------------------------------

def run_differential_evolution(
    objective_fn: Callable[[np.ndarray], np.ndarray],
    bounds: Sequence[tuple],
    directions: Sequence[str],
    n_weight_sets: int = 15,
    maxiter: int = 150,
    seed: int = 42,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> OptimizationResult:
    """Multi-objective search via Differential Evolution + weighted scalarization.

    For a single objective this is just plain DE. For 2+ objectives, it solves
    a series of weighted-sum sub-problems (weights spanning the simplex) with
    `scipy.optimize.differential_evolution`, then keeps the non-dominated
    union of all the per-weight optima. This tends to converge faster and more
    precisely than a genetic algorithm on smooth, continuous surrogates (e.g.
    Ridge/GPR/SVR), at the cost of being less robust on very rugged/discrete
    objective landscapes (e.g. deep tree ensembles) than NSGA-II.
    """
    from scipy.optimize import differential_evolution

    rng = np.random.default_rng(seed)
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    sign = np.array([1.0 if dirn == "min" else -1.0 for dirn in directions])
    n_obj = len(directions)

    weight_sets = _make_weight_sets(n_obj, n_weight_sets, rng)
    obj_lo, obj_span = _normalize_probe(objective_fn, lb, ub, sign, rng, n_probe=min(60, max(20, 4 * len(bounds))))

    all_X, all_F = [], []
    total = len(weight_sets)
    for idx, weights in enumerate(weight_sets):
        def scalarized(x, weights=weights):
            raw = np.asarray(objective_fn(np.atleast_2d(x)), dtype=float)[0]
            corrected = raw * sign
            normalized = (corrected - obj_lo) / obj_span
            return float(np.dot(weights, normalized))

        result = differential_evolution(
            scalarized,
            bounds=list(zip(lb, ub)),
            maxiter=maxiter,
            seed=int(rng.integers(0, 1_000_000)),
            polish=True,
            tol=1e-7,
            updating="deferred",
        )
        x_best = result.x
        f_best = np.asarray(objective_fn(np.atleast_2d(x_best)), dtype=float)[0]
        all_X.append(x_best)
        all_F.append(f_best)
        if progress_callback is not None:
            progress_callback(idx + 1, total)

    X = np.array(all_X)
    F_natural = np.array(all_F)
    F_min = F_natural * sign
    front_idx = pareto_front(F_min)
    return OptimizationResult(X=X[front_idx], F=F_natural[front_idx])


# ---------------------------------------------------------------------------
# MODEL-BASED / HYBRID: surrogate-assisted evolutionary optimization
# (RBFMOpt-style with an RBF interpolant, or ANN-assisted with an MLP)
# ---------------------------------------------------------------------------

def _latin_hypercube_unit(n_samples: int, d: int, rng: np.random.Generator) -> np.ndarray:
    cut = np.linspace(0.0, 1.0, n_samples + 1)
    a, b = cut[:n_samples], cut[1:n_samples + 1]
    pts = np.zeros((n_samples, d))
    for j in range(d):
        perm = rng.permutation(n_samples)
        pts[:, j] = a[perm] + rng.random(n_samples) * (b[perm] - a[perm])
    return pts


def _run_surrogate_assisted(
    objective_fn: Callable[[np.ndarray], np.ndarray],
    bounds: Sequence[tuple],
    directions: Sequence[str],
    inner_surrogate: str,
    initial_samples: int,
    refine_iterations: int,
    candidates_per_iteration: int,
    inner_pop_size: int,
    inner_n_gen: int,
    seed: int,
    progress_callback: Optional[Callable[[int, int], None]],
) -> OptimizationResult:
    rng = np.random.default_rng(seed)
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    d = len(bounds)
    n_obj = len(directions)
    sign = np.array([1.0 if dirn == "min" else -1.0 for dirn in directions])

    unit_pts = _latin_hypercube_unit(initial_samples, d, rng)
    X_archive = lb + unit_pts * (ub - lb)
    F_archive = np.atleast_2d(np.asarray(objective_fn(X_archive), dtype=float))

    for it in range(refine_iterations):
        if inner_surrogate == "rbf":
            from scipy.interpolate import RBFInterpolator

            meta_model = RBFInterpolator(X_archive, F_archive, kernel="thin_plate_spline", smoothing=1e-6)

            def meta_objective(Xq):
                pred = np.atleast_2d(meta_model(Xq))
                return pred
        else:  # "ann"
            from sklearn.neural_network import MLPRegressor

            meta_model = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=3000, random_state=int(rng.integers(0, 1_000_000)))
            meta_model.fit(X_archive, F_archive if n_obj > 1 else F_archive[:, 0])

            def meta_objective(Xq):
                pred = np.asarray(meta_model.predict(Xq), dtype=float)
                return pred if pred.ndim == 2 else pred.reshape(-1, 1)

        inner_result = run_nsga2(
            meta_objective, bounds, directions,
            pop_size=inner_pop_size, n_gen=inner_n_gen,
            seed=int(rng.integers(0, 1_000_000)),
        )

        n_pick = min(candidates_per_iteration, len(inner_result.X))
        if n_pick == 0:
            X_new = lb + rng.random((candidates_per_iteration, d)) * (ub - lb)
        else:
            pick_idx = rng.choice(len(inner_result.X), size=n_pick, replace=False)
            X_new = inner_result.X[pick_idx]

        F_new = np.atleast_2d(np.asarray(objective_fn(X_new), dtype=float))
        X_archive = np.vstack([X_archive, X_new])
        F_archive = np.vstack([F_archive, F_new])

        if progress_callback is not None:
            progress_callback(it + 1, refine_iterations)

    F_min = F_archive * sign
    front_idx = pareto_front(F_min)
    return OptimizationResult(X=X_archive[front_idx], F=F_archive[front_idx])


def run_rbf_surrogate_assisted(
    objective_fn: Callable[[np.ndarray], np.ndarray],
    bounds: Sequence[tuple],
    directions: Sequence[str],
    initial_samples: int = 50,
    refine_iterations: int = 5,
    candidates_per_iteration: int = 15,
    inner_pop_size: int = 60,
    inner_n_gen: int = 40,
    seed: int = 42,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> OptimizationResult:
    """RBFMOpt-style surrogate-assisted optimization: fit a cheap Radial
    Basis Function interpolant on all points evaluated so far, run NSGA-II
    on THAT cheap interpolant to propose promising candidates, evaluate the
    real objective only on those candidates, and repeat. This is the classic
    "surrogate-based optimization" (SBO) loop from the RBFOpt/pySOT family —
    it minimizes the number of calls to the (potentially expensive) real
    objective function, at the cost of some solution quality versus a direct
    metaheuristic search when the real objective is actually cheap (as it is
    here, since it's a trained ML surrogate rather than a physical
    simulation) — its main value shows up if you later swap in a genuinely
    expensive simulator as the objective.
    """
    return _run_surrogate_assisted(
        objective_fn, bounds, directions, "rbf",
        initial_samples, refine_iterations, candidates_per_iteration,
        inner_pop_size, inner_n_gen, seed, progress_callback,
    )


def run_ann_surrogate_assisted(
    objective_fn: Callable[[np.ndarray], np.ndarray],
    bounds: Sequence[tuple],
    directions: Sequence[str],
    initial_samples: int = 50,
    refine_iterations: int = 5,
    candidates_per_iteration: int = 15,
    inner_pop_size: int = 60,
    inner_n_gen: int = 40,
    seed: int = 42,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> OptimizationResult:
    """Same surrogate-assisted loop as `run_rbf_surrogate_assisted`, but the
    cheap internal meta-surrogate is a small MLP (ANN) instead of an RBF
    interpolant. ANNs can capture more complex, non-smooth response surfaces
    than RBF given enough archive points, but need more data to be reliable
    and are more sensitive to the initial sample size than RBF is.

    (Note: a literal CNN-based surrogate is intentionally not offered here.
    Convolutional layers exploit spatial/grid locality, which plain tabular
    design variables like these don't have — a CNN over a flat feature
    vector wouldn't behave meaningfully differently from this MLP. CNN
    surrogates are genuinely useful when design variables ARE spatial, e.g.
    pixel-grid facade layouts or floor plans; happy to add that mode if your
    variables take that form.)
    """
    return _run_surrogate_assisted(
        objective_fn, bounds, directions, "ann",
        initial_samples, refine_iterations, candidates_per_iteration,
        inner_pop_size, inner_n_gen, seed, progress_callback,
    )
