#!/usr/bin/env python3
"""Deterministically check the worked examples accompanying Homework 4.

Run: python3 verify.py
No third-party packages are required. Results are written next to this script.
These finite numerical examples are sanity checks, not proofs of the theorems.
The vertex enumerator below is only used for the particular small LPs in these
checks; it is not a general LP solver or an unboundedness detector.
"""

from itertools import combinations
import json
from pathlib import Path
import random
import sys


TOL = 1e-9
SEED = 20260927


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, message):
    require(abs(actual - expected) <= TOL, f"{message}: {actual} != {expected}")


def dot(a, b):
    require(len(a) == len(b), "Dimension mismatch in dot product")
    return sum(u * v for u, v in zip(a, b))


def matvec(matrix, vector):
    return [dot(row, vector) for row in matrix]


def transpose(matrix):
    return [list(column) for column in zip(*matrix)]


def solve_square(matrix, rhs):
    """Small Gaussian elimination with partial pivoting; None if singular."""
    n = len(rhs)
    rows = [[float(v) for v in row] + [float(value)]
            for row, value in zip(matrix, rhs)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda r: abs(rows[r][column]))
        if abs(rows[pivot][column]) <= TOL:
            return None
        rows[column], rows[pivot] = rows[pivot], rows[column]
        divisor = rows[column][column]
        rows[column] = [v / divisor for v in rows[column]]
        for row in range(n):
            if row != column:
                scale = rows[row][column]
                rows[row] = [u - scale * v
                             for u, v in zip(rows[row], rows[column])]
    return [row[-1] for row in rows]


def vertices(matrix, rhs, dimension):
    """Enumerate feasible intersections of dimension rows of M z <= rhs."""
    found = []
    for active in combinations(range(len(rhs)), dimension):
        point = solve_square([matrix[i] for i in active], [rhs[i] for i in active])
        if point is None:
            continue
        if not all(dot(row, point) <= bound + TOL for row, bound in zip(matrix, rhs)):
            continue
        if not any(max(abs(a - b) for a, b in zip(point, prior)) <= TOL
                   for prior in found):
            found.append(point)
    require(bool(found), "This example was expected to have a feasible vertex")
    return found


def vertex_optimum(objective, matrix, rhs, maximize=False):
    candidates = vertices(matrix, rhs, len(objective))
    pick = max if maximize else min
    point = pick(candidates, key=lambda v: dot(objective, v))
    return dot(objective, point), point


def check_stationary_distribution():
    matrix = [[0.9, 0.1], [0.4, 0.6]]
    pi = [0.8, 0.2]
    for row in matrix:
        close(sum(row), 1.0, "Transition row sum")
        require(all(value >= 0 for value in row), "Transition entries must be nonnegative")
    close(sum(pi), 1.0, "Probability normalization")
    require(all(value >= 0 for value in pi), "Probability entries must be nonnegative")
    residual = max(abs(a - b) for a, b in zip(matvec(transpose(matrix), pi), pi))
    close(residual, 0.0, "Stationarity residual")
    return {"P": matrix, "pi": pi, "max_stationarity_residual": residual}


def check_swap_cycle():
    matrix = [[0.0, 1.0], [1.0, 0.0]]
    current = [1.0, 0.0]
    sequence = [current]
    for _ in range(6):
        current = matvec(transpose(matrix), current)
        sequence.append(current)
    for i, state in enumerate(sequence):
        expected = [1.0, 0.0] if i % 2 == 0 else [0.0, 1.0]
        require(state == expected, "Swap chain must alternate exactly")
    require(matvec(transpose(matrix), [0.5, 0.5]) == [0.5, 0.5],
            "Swap chain invariant distribution")
    for basis in [[1.0, 0.0], [0.0, 1.0]]:
        require(matvec(matrix, matvec(matrix, basis)) == basis, "Swap squared must be identity")
    return {"P": matrix, "invariant_distribution": [0.5, 0.5],
            "initial_state": sequence[0], "first_seven_distributions": sequence,
            "interpretation": "An exact two-cycle; stationarity does not imply convergence from every initial distribution."}


def check_piecewise_linear():
    slopes = [2.0, -1.0, 1.0]
    offsets = [0.0, -3.0, 1.0]  # f(x) = max_i (slopes[i] * x - offsets[i]).
    primal_value, primal_point = vertex_optimum([0.0, 1.0],
                                               [[a, -1.0] for a in slopes], offsets)
    dual_matrix = [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0],
                   slopes, [-a for a in slopes], [1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]
    dual_rhs = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0]
    dual_value, dual_point = vertex_optimum([-b for b in offsets], dual_matrix,
                                           dual_rhs, maximize=True)
    expected_p = [1.0 / 3.0, 2.0 / 3.0, 0.0]
    close(primal_point[0], 1.0, "Primal optimizer")
    close(primal_value, 2.0, "Primal optimal value")
    close(dual_value, primal_value, "Primal-dual gap")
    for actual, expected in zip(dual_point, expected_p):
        close(actual, expected, "Dual optimizer component")
    close(dot(slopes, expected_p), 0.0, "Dual weighted slope")
    close(sum(expected_p), 1.0, "Dual normalization")
    gaps = [primal_value - (a * primal_point[0] - b) for a, b in zip(slopes, offsets)]
    for weight, gap in zip(expected_p, gaps):
        close(weight * gap, 0.0, "Complementary slackness")
    for x in [-100.0, -3.0, 0.0, 0.9, 1.0, 1.1, 7.0, 100.0]:
        require(max(a * x - b for a, b in zip(slopes, offsets)) >= dual_value - TOL,
                "Dual value must lower-bound sampled primal objectives")
    return {"slopes": slopes, "b": offsets, "x_star": primal_point[0],
            "primal_value": primal_value, "dual_value": dual_value,
            "dual_p": dual_point, "affine_piece_gaps": gaps}


def check_symmetric_indefinite():
    matrix = [[0.0, 1.0], [1.0, 0.0]]
    cost = [1.0, 1.0]
    x_star = [1.0, 1.0]
    require(matrix == transpose(matrix), "Matrix must be symmetric")
    require(matvec(matrix, x_star) == cost, "Equality certificate")
    positive_form = dot([1.0, 1.0], matvec(matrix, [1.0, 1.0]))
    negative_form = dot([1.0, -1.0], matvec(matrix, [1.0, -1.0]))
    require(positive_form > 0 and negative_form < 0, "Matrix must be indefinite")
    value, optimizer = vertex_optimum(cost, [[-v for v in row] for row in matrix]
                                     + [[-1.0, 0.0], [0.0, -1.0]], [-v for v in cost] + [0.0, 0.0])
    close(value, dot(cost, x_star), "Symmetric primal-dual certificate value")
    return {"A": matrix, "c": cost, "x_star": x_star, "lp_optimizer": optimizer,
            "optimal_value": value, "positive_quadratic_form": positive_form,
            "negative_quadratic_form": negative_form}


def check_nonsymmetric_counterexample():
    matrix = [[1.0, 1.0], [0.0, 1.0]]
    cost = [2.0, 1.0]
    equality_point = [1.0, 1.0]
    better_point = [0.0, 2.0]
    require(matrix != transpose(matrix), "Counterexample matrix must be nonsymmetric")
    require(matvec(matrix, equality_point) == cost, "Counterexample equality point")
    require(all(a >= b for a, b in zip(matvec(matrix, better_point), cost)),
            "Better point must satisfy Ax >= c")
    require(all(v >= 0 for v in better_point), "Better point must be nonnegative")
    value, optimizer = vertex_optimum(cost, [[-v for v in row] for row in matrix]
                                     + [[-1.0, 0.0], [0.0, -1.0]], [-v for v in cost] + [0.0, 0.0])
    close(dot(cost, equality_point), 3.0, "Equality point cost")
    close(dot(cost, better_point), 2.0, "Better feasible point cost")
    close(value, 2.0, "Counterexample optimal value")
    require(any(a > b + TOL for a, b in zip(matvec(transpose(matrix), equality_point), cost)),
            "Equality point must fail dual feasibility")
    return {"A": matrix, "c": cost, "equality_point": equality_point,
            "equality_point_cost": dot(cost, equality_point), "better_feasible_point": better_point,
            "better_point_cost": dot(cost, better_point), "lp_optimizer": optimizer,
            "lp_optimal_value": value, "A_transpose_times_equality_point": matvec(transpose(matrix), equality_point)}


def check_robust_simplexes():
    rng = random.Random(SEED)
    directions = [[0.0, 0.0], [2.0, 3.0], [-2.0, -3.0], [2.0, -3.0], [-2.0, 3.0],
                  [-7.0, 0.0], [0.0, -7.0]]
    directions += [[rng.uniform(-10.0, 10.0), rng.uniform(-10.0, 10.0)] for _ in range(64)]
    u2_matrix = [[-1.0, 0.0], [0.0, -1.0], [1.0, 1.0], [-1.0, -1.0]]
    u2_rhs = [0.0, 0.0, 1.0, -1.0]
    u3_matrix = [[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]
    u3_rhs = [0.0, 0.0, 1.0]
    max_gaps = {"U2": 0.0, "U3": 0.0}
    for x in directions:
        primal2, a2 = vertex_optimum(x, u2_matrix, u2_rhs, maximize=True)
        dual2, y = vertex_optimum([1.0], [[-1.0], [-1.0]], [-x[0], -x[1]])
        close(primal2, max(x), "Simplex support function")
        close(dual2, primal2, "U2 primal-dual gap")
        require(all(y[0] >= value - TOL for value in x), "U2 dual certificate feasibility")
        primal3, a3 = vertex_optimum(x, u3_matrix, u3_rhs, maximize=True)
        dual3, mu = vertex_optimum([1.0], [[-1.0], [-1.0], [-1.0]], [0.0, x[0], x[1]])
        close(primal3, max(0.0, -x[0], -x[1]), "Negative simplex support function")
        close(dual3, primal3, "U3 primal-dual gap")
        require(mu[0] >= -TOL and all(mu[0] >= -value - TOL for value in x),
                "U3 dual certificate feasibility")
        for _ in range(8):
            weight = rng.random()
            sample2 = [weight, 1.0 - weight]
            radius = rng.random()
            sample3 = [-radius * weight, -radius * (1.0 - weight)]
            require(dot(x, sample2) <= dual2 + TOL, "U2 bound on interior sample")
            require(dot(x, sample3) <= dual3 + TOL, "U3 bound on interior sample")
        require(dot(x, a2) > primal2 - 0.1, "Tightening U2 RHS must fail at a worst-case vertex")
        require(dot(x, a3) > primal3 - 0.1, "Tightening U3 RHS must fail at a worst-case vertex")
        max_gaps["U2"] = max(max_gaps["U2"], abs(primal2 - dual2))
        max_gaps["U3"] = max(max_gaps["U3"], abs(primal3 - dual3))
    return {"seed": SEED, "directions_checked": len(directions),
            "interior_samples_per_set_per_direction": 8, "maximum_primal_dual_gaps": max_gaps,
            "U2_vertices": vertices(u2_matrix, u2_rhs, 2), "U3_vertices": vertices(u3_matrix, u3_rhs, 2),
            "U2_dual": "min y subject to y >= x1, y >= x2; y unrestricted",
            "U3_dual": "min mu subject to mu >= 0, mu >= -x1, mu >= -x2"}


def check_unbounded_uncertainty():
    # U2 = {(1,t): t >= 0}; r=(0,1) is a feasible recession direction.
    base = [1.0, 0.0]
    ray = [0.0, 1.0]
    require(dot([1.0, 0.0], ray) == 0.0 and all(v >= 0 for v in ray),
            "Recession direction must preserve D a = d and nonnegativity")
    results = []
    for x in [[2.0, 1.0], [-3.0, 0.2], [4.0, 0.0], [-2.0, -0.5], [2.0, -3.0]]:
        slope = dot(x, ray)
        if slope > 0:
            require(0.0 < x[1], "Dual requirement 0 >= x2 must be infeasible")
            for increase in [1.0, 100.0, 10000.0]:
                t = increase / slope
                close(dot(x, [1.0, t]) - dot(x, base), increase, "Unbounded objective growth along ray")
            results.append({"x": x, "support": "+infinity", "positive_ray_slope": slope,
                            "dual_feasible": False})
        else:
            y = x[0]
            require(y >= x[0] - TOL and 0.0 >= x[1], "Finite-support dual certificate")
            for t in [0.0, 1.0, 10.0, 1000000.0]:
                require(dot(x, [1.0, t]) <= y + TOL, "Unbounded set with finite support")
            close(dot(x, base), y, "Finite support is attained")
            results.append({"x": x, "support": y, "dual_y": y, "dual_feasible": True})
    return {"uncertainty_set": "a1 = 1, a2 >= 0", "recession_direction": ray, "cases": results}


def main():
    checks = [check_stationary_distribution, check_swap_cycle, check_piecewise_linear,
              check_symmetric_indefinite, check_nonsymmetric_counterexample,
              check_robust_simplexes, check_unbounded_uncertainty]
    results = []
    for check in checks:
        try:
            details = check()
            results.append({"name": check.__name__, "status": "passed", "details": details})
        except Exception as error:
            results.append({"name": check.__name__, "status": "failed", "error": str(error)})
    passed = sum(result["status"] == "passed" for result in results)
    report = {"status": "passed" if passed == len(checks) else "failed",
              "scope": "Deterministic numerical sanity checks of guide examples; not proofs of the general theorems.",
              "method": "Direct algebra, low-dimensional feasible-vertex enumeration, primal-dual certificates, and explicit recession rays.",
              "solver_limit": "The vertex enumerator is not a general LP solver and does not detect unbounded objectives.",
              "third_party_dependencies": [], "absolute_tolerance": TOL, "seed": SEED,
              "checks_passed": passed, "checks_total": len(checks), "checks": results}
    destination = Path(__file__).resolve().with_name("verification_results.json")
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{passed}/{len(checks)} example checks passed; wrote {destination.name}")
    for result in results:
        print(f"  {result['status']}: {result['name']}")
        if result["status"] == "failed":
            print(f"    {result['error']}")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
