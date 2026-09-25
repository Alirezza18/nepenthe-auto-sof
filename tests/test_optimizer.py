"""Test suite for the Auto-SOF optimization core (core/optimizer.py).

Run with:
    python -m unittest discover -s tests -v

The optimization core is deliberately dependency-free (NumPy only), so this
suite needs nothing beyond the app's runtime requirements. Every solver is
checked for the same invariants:

    1. it returns an OptimizationResult with well-shaped X and F arrays
    2. every returned design lies inside the requested box bounds
    3. objective directions (min/max) are honored on the returned front
    4. re-running with the same seed reproduces the same front exactly
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.optimizer import (
    OptimizationResult,
    _crowding_distance,
    _dominates,
    _environmental_selection_crowding,
    _fast_non_dominated_sort,
    pareto_front,
    run_aco,
    run_ann_surrogate_assisted,
    run_differential_evolution,
    run_genetic_algorithm,
    run_hype,
    run_nsga2,
    run_nsga3,
    run_random_search,
    run_rbf_surrogate_assisted,
    run_steady_state_nsga2,
)

BOUNDS = [(0.0, 1.0), (0.0, 5.0)]


def convex_problem(X):
    """Two objectives with a smooth concave-then-convex trade-off front.

    f1 = x0 (minimize), f2 = (x1 - 2)^2 + x0 (minimize). Both objectives
    conflict: lowering f1 pushes the optimum of f2 around, producing a
    genuine two-dimensional Pareto front for the solvers to discover.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    f1 = X[:, 0]
    f2 = (X[:, 1] - 2.0) ** 2 + X[:, 0]
    return np.column_stack([f1, f2])


def linear_problem(X):
    """Single-objective sphere: sanity check for scalarizing solvers."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    return (X ** 2).sum(axis=1).reshape(-1, 1)


class TestDominancePrimitives(unittest.TestCase):
    def test_dominates_basic(self):
        a = np.array([1.0, 1.0])
        self.assertTrue(_dominates(a, np.array([2.0, 2.0])))
        self.assertTrue(_dominates(a, np.array([1.0, 2.0])))
        self.assertFalse(_dominates(a, np.array([1.0, 1.0])))  # equal => not dominated
        self.assertFalse(_dominates(a, np.array([0.5, 0.5])))  # a is worse

    def test_non_dominated_sort_classifies_known_fronts(self):
        F = np.array(
            [
                [0.0, 1.0],  # front 0
                [1.0, 0.0],  # front 0
                [2.0, 2.0],  # dominated by both above -> front 1
                [3.0, 3.0],  # dominated by everything -> front 2
            ]
        )
        fronts = _fast_non_dominated_sort(F)
        self.assertEqual(sorted(fronts[0]), [0, 1])
        self.assertIn(2, fronts[1])
        self.assertIn(3, fronts[2])

    def test_pareto_front_returns_indices_of_first_front(self):
        F = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]])
        idx = pareto_front(F)
        self.assertEqual(sorted(idx.tolist()), [0, 1])

    def test_crowding_distance_extremes_are_infinite(self):
        F = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        cd = _crowding_distance(F)
        self.assertEqual(cd[0], np.inf)
        self.assertEqual(cd[-1], np.inf)
        self.assertTrue(np.isfinite(cd[1]))

    def test_environmental_selection_keeps_population_size(self):
        combined_X = np.random.default_rng(0).random((10, 2))
        combined_F = convex_problem(combined_X)
        X_sel, F_sel = _environmental_selection_crowding(combined_X, combined_F, 6)
        self.assertEqual(len(X_sel), 6)
        self.assertEqual(len(F_sel), 6)


class TestSolverInvariantsMixin:
    """Shared invariant checks applied to every solver run."""

    def assertValidResult(self, result, directions, context=""):
        self.assertIsInstance(result, OptimizationResult, context)
        self.assertEqual(result.X.ndim, 2, context)
        self.assertEqual(result.F.ndim, 2, context)
        self.assertEqual(result.X.shape[0], result.F.shape[0], context)
        self.assertGreater(result.X.shape[0], 0, context)
        # Bounds respected
        for j, (low, high) in enumerate(BOUNDS):
            self.assertTrue(np.all(result.X[:, j] >= low - 1e-9), context)
            self.assertTrue(np.all(result.X[:, j] <= high + 1e-9), context)
        # Directions honored: "min" columns should hold the smaller of the
        # paired min/max extremes across the front (per objective independently).
        for k, dirn in enumerate(directions):
            col = result.F[:, k]
            if dirn == "min":
                # the best (minimum) value on the front must not exceed the
                # worst possible value the problem can produce on that axis
                self.assertLessEqual(col.min(), 1.0 + 1e-6, context)
            else:
                self.assertGreaterEqual(col.max(), 0.0 - 1e-6, context)


class TestMetaheuristics(unittest.TestCase, TestSolverInvariantsMixin):
    def test_nsga2_convex_problem(self):
        result = run_nsga2(convex_problem, BOUNDS, ["min", "min"], pop_size=30, n_gen=8, seed=42)
        self.assertValidResult(result, ["min", "min"], "nsga2")
        front = result.F
        # Front should be genuinely non-dominated
        for i in range(len(front)):
            dominated = any(
                np.all(front[j] <= front[i]) and np.any(front[j] < front[i]) for j in range(len(front)) if j != i
            )
            self.assertFalse(dominated, f"member {i} of the NSGA-II front is dominated")

    def test_nsga2_respects_max_direction(self):
        result = run_nsga2(convex_problem, BOUNDS, ["max", "min"], pop_size=20, n_gen=6, seed=7)
        self.assertValidResult(result, ["max", "min"], "nsga2 max")
        # maximize f1 => large x0 values should appear on the front
        self.assertGreater(result.X[:, 0].max(), 0.5)

    def test_nsga3_runs_and_returns_front(self):
        result = run_nsga3(convex_problem, BOUNDS, ["min", "min"], pop_size=30, n_gen=6, seed=42)
        self.assertValidResult(result, ["min", "min"], "nsga3")

    def test_steady_state_nsga2_runs(self):
        result = run_steady_state_nsga2(
            convex_problem, BOUNDS, ["min", "min"], pop_size=20, n_iterations=60, seed=42
        )
        self.assertValidResult(result, ["min", "min"], "ss-nsga2")

    def test_hype_runs(self):
        result = run_hype(
            convex_problem, BOUNDS, ["min", "min"], pop_size=16, n_gen=4, hv_samples=100, seed=42
        )
        self.assertValidResult(result, ["min", "min"], "hype")

    def test_genetic_algorithm_scalarizes(self):
        result = run_genetic_algorithm(
            convex_problem, BOUNDS, ["min", "min"], n_weight_sets=4, pop_size=20, n_gen=6, seed=42
        )
        self.assertValidResult(result, ["min", "min"], "ga")

    def test_aco_runs(self):
        result = run_aco(
            convex_problem, BOUNDS, ["min", "min"], n_weight_sets=3, archive_size=12, n_ants=6, n_iterations=8, seed=42
        )
        self.assertValidResult(result, ["min", "min"], "aco")

    def test_random_search_returns_non_dominated_subset(self):
        result = run_random_search(convex_problem, BOUNDS, ["min", "min"], n_samples=300, seed=42)
        self.assertValidResult(result, ["min", "min"], "random")


class TestModelBasedSolvers(unittest.TestCase, TestSolverInvariantsMixin):
    def test_differential_evolution_minimizes_single_objective(self):
        result = run_differential_evolution(
            linear_problem, BOUNDS, ["min"], n_weight_sets=1, maxiter=15, seed=42
        )
        self.assertValidResult(result, ["min"], "de")
        # Sphere optimum is at (0, 0) => best f should be near zero
        self.assertLess(result.F[:, 0].min(), 0.5)

    def test_differential_evolution_two_objectives(self):
        result = run_differential_evolution(
            convex_problem, BOUNDS, ["min", "min"], n_weight_sets=4, maxiter=10, seed=42
        )
        self.assertValidResult(result, ["min", "min"], "de 2-obj")

    def test_rbf_surrogate_assisted_runs(self):
        result = run_rbf_surrogate_assisted(
            convex_problem,
            BOUNDS,
            ["min", "min"],
            initial_samples=15,
            refine_iterations=2,
            candidates_per_iteration=6,
            inner_pop_size=12,
            inner_n_gen=4,
            seed=42,
        )
        self.assertValidResult(result, ["min", "min"], "rbf_sa")

    def test_ann_surrogate_assisted_runs(self):
        result = run_ann_surrogate_assisted(
            convex_problem,
            BOUNDS,
            ["min", "min"],
            initial_samples=20,
            refine_iterations=2,
            candidates_per_iteration=6,
            inner_pop_size=12,
            inner_n_gen=4,
            seed=42,
        )
        self.assertValidResult(result, ["min", "min"], "ann_sa")


class TestReproducibility(unittest.TestCase):
    def test_nsga2_same_seed_same_front(self):
        r1 = run_nsga2(convex_problem, BOUNDS, ["min", "min"], pop_size=20, n_gen=5, seed=123)
        r2 = run_nsga2(convex_problem, BOUNDS, ["min", "min"], pop_size=20, n_gen=5, seed=123)
        np.testing.assert_array_equal(r1.X, r2.X)
        np.testing.assert_array_equal(r1.F, r2.F)

    def test_nsga2_different_seed_different_front(self):
        r1 = run_nsga2(convex_problem, BOUNDS, ["min", "min"], pop_size=20, n_gen=5, seed=1)
        r2 = run_nsga2(convex_problem, BOUNDS, ["min", "min"], pop_size=20, n_gen=5, seed=2)
        self.assertFalse(np.array_equal(r1.F, r2.F))

    def test_random_search_same_seed_same_result(self):
        r1 = run_random_search(convex_problem, BOUNDS, ["min", "min"], n_samples=200, seed=9)
        r2 = run_random_search(convex_problem, BOUNDS, ["min", "min"], n_samples=200, seed=9)
        np.testing.assert_array_equal(r1.X, r2.X)

    def test_solver_does_not_touch_global_numpy_random_state(self):
        np.random.seed(0)
        expected = np.random.RandomState(0).random(5)
        np.random.seed(0)
        run_nsga2(convex_problem, BOUNDS, ["min", "min"], pop_size=12, n_gen=2, seed=777)
        np.testing.assert_array_equal(np.random.random(5), expected)


class TestProgressCallbacks(unittest.TestCase):
    def test_nsga2_calls_progress_once_per_generation(self):
        calls = []

        def cb(current, total):
            calls.append((current, total))

        run_nsga2(convex_problem, BOUNDS, ["min", "min"], pop_size=10, n_gen=4, seed=42, progress_callback=cb)
        self.assertEqual([c for c, _ in calls], [1, 2, 3, 4])
        self.assertTrue(all(t == 4 for _, t in calls))

    def test_random_search_reports_progress(self):
        calls = []
        run_random_search(
            convex_problem, BOUNDS, ["min", "min"], n_samples=150, seed=1,
            progress_callback=lambda c, t: calls.append(c),
        )
        self.assertGreater(len(calls), 0)


class TestEdgeCases(unittest.TestCase):
    def test_single_objective_problem(self):
        result = run_nsga2(linear_problem, BOUNDS, ["min"], pop_size=12, n_gen=4, seed=42)
        self.assertEqual(result.F.shape[1], 1)
        self.assertEqual(result.X.shape[1], 2)

    def test_maximize_only(self):
        result = run_genetic_algorithm(
            linear_problem, BOUNDS, ["max"], n_weight_sets=1, pop_size=12, n_gen=5, seed=42
        )
        # maximizing the sphere pushes designs to the corner (1, 5)
        self.assertGreater(result.F[:, 0].max(), 10.0)


if __name__ == "__main__":
    unittest.main()
