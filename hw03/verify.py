"""Numerical checks for HW3: (1) primal/dual pair, (2) theorem of the alternative, (3) 101-subset result."""
import numpy as np
from scipy.optimize import linprog

rng = np.random.default_rng(0)
INF = None  # linprog's notation for an infinite bound
free = lambda k: [(INF, INF)] * k

# ------------------------------------------------------------------ Problem 1
print("=== Problem 1: primal and derived dual ===")
c = np.array([1.0, -1, 0, 0])
A = np.array([[2.0, 3, -1, 1], [3, 1, 4, -2], [-1, -1, 2, 1]])
b = np.array([0.0, 3, 6])
# primal: row1 <=, row2 >= (negated), row3 =;  x1<=0, x2,x3>=0, x4 free
P = linprog(c, A_ub=np.vstack([A[0], -A[1]]), b_ub=[b[0], -b[1]], A_eq=A[2:3], b_eq=b[2:3],
            bounds=[(INF, 0), (0, INF), (0, INF), (INF, INF)], method="highs")
print("primal LP:", P.message.split(".")[0])
# dual: max 3p2+6p3  s.t. A1'p>=1, A2'p<=-1, A3'p<=0, A4'p=0, p1<=0, p2>=0, p3 free
D = linprog(-b, A_ub=np.vstack([-A[:, 0], A[:, 1], A[:, 2]]), b_ub=[-1, -1, 0],
            A_eq=A[:, 3][None, :], b_eq=[0], bounds=[(INF, 0), (0, INF), (INF, INF)], method="highs")
print("dual LP  :", D.message.split(".")[0])
x0, d = np.array([0.0, 0, 2, 2]), np.array([-1.0, 0, 1, -3])
for name, v, rhs in [("x0 ", x0, b), ("ray", d, 0 * b)]:
    r = A @ v
    ok = r[0] <= rhs[0] and r[1] >= rhs[1] and r[2] == rhs[2] and v[0] <= 0 and v[1] >= 0 and v[2] >= 0
    print(f"{name}= {v}:  A{name.strip()} = {r},  c'{name.strip()} = {c @ v:+.0f},  satisfies constraints: {ok}")

# ------------------------------------------------------------------ Problem 2
print("\n=== Problem 2: (i) <=> (ii) on 2000 random integer matrices ===")
def stmt_i(M):   # max x1 s.t. Mx>=0, x>=0, x1<=1 (the box keeps the LP bounded); (i) holds iff optimum is 0
    n = M.shape[1]; obj = np.zeros(n); obj[0] = -1
    r = linprog(obj, A_ub=-M, b_ub=np.zeros(M.shape[0]), bounds=[(0, 1)] + [(0, INF)] * (n - 1), method="highs")
    return abs(r.fun) < 1e-9
def stmt_ii(M):  # find p>=0 with p'M<=0 and p'M1<=-1 (normalised form of the strict inequality)
    m, n = M.shape; rhs = np.zeros(n); rhs[0] = -1
    return linprog(np.zeros(m), A_ub=M.T, b_ub=rhs, bounds=[(0, INF)] * m, method="highs").status == 0
counts = {}
for _ in range(2000):
    M = rng.integers(-3, 4, size=(rng.integers(2, 7), rng.integers(2, 6))).astype(float)
    key = (stmt_i(M), stmt_ii(M)); counts[key] = counts.get(key, 0) + 1
print("both true:", counts.get((True, True), 0), "| both false:", counts.get((False, False), 0),
      "| disagreements:", counts.get((True, False), 0) + counts.get((False, True), 0))

# ------------------------------------------------------------------ Problem 3
print("\n=== Problem 3: 500 inequalities in 100 variables ===")
m, n = 500, 100
A3 = rng.standard_normal((m, n)); b3 = rng.standard_normal(m)
def infeasible(rows):
    """Phase-1 test: t* = min_x max_i (a_i'x - b_i), clipped below at -1.  Infeasible <=> t* > 0.
    (A zero-objective feasibility LP is avoided: HiGHS can return 'status unknown' on it.)"""
    k = len(rows)
    r = linprog(np.r_[np.zeros(n), 1.0], A_ub=np.hstack([A3[rows], -np.ones((k, 1))]), b_ub=b3[rows],
                bounds=free(n) + [(-1, INF)], method="highs")
    assert r.status == 0, r.message
    return r.fun > 1e-7
print("full system infeasible:", infeasible(np.arange(m)))

# (a) duality route.  Dual of  min t s.t. a_i'x - t <= b_i  is  max -b'p s.t. A'p=0, 1'p=1, p>=0  (101 equalities)
Aeq = np.vstack([A3.T, np.ones((1, m))]); beq = np.r_[np.zeros(n), 1.0]
dual = linprog(b3, A_eq=Aeq, b_eq=beq, bounds=[(0, INF)] * m, method="highs-ds")   # simplex => a vertex
p = dual.x; S = np.flatnonzero(p > 1e-9)
print(f"(a) dual optimum t* = {-dual.fun:.4f} > 0;  vertex support |S| = {len(S)}  (bound n+1 = {n + 1})")
print(f"    certificate: ||sum p_i a_i|| = {np.linalg.norm(A3[S].T @ p[S]):.1e},  sum p_i b_i = {p[S] @ b3[S]:.4f} < 0")
print("    subsystem S infeasible (independent LP check):", infeasible(S))
drop = sum(infeasible(np.delete(S, k)) for k in range(len(S)))
print(f"    dropping any single row of S: still infeasible in {drop} of {len(S)} cases (so S is minimal)")

# (b) constructive route: start from a DENSE Farkas certificate and shrink its support (Caratheodory-style)
verts = [linprog(b3 + 0.3 * rng.standard_normal(m), A_eq=Aeq, b_eq=beq, bounds=[(0, INF)] * m, method="highs-ds").x
         for _ in range(6)]
q = np.mean(verts, axis=0); q /= -(b3 @ q)                     # normalise so that b'q = -1
print(f"(b) dense certificate: support {np.sum(q > 1e-12)},  ||A'q|| = {np.linalg.norm(A3.T @ q):.1e},  b'q = {b3 @ q:.4f}")
history = [int(np.sum(q > 1e-12))]
while True:
    T = np.flatnonzero(q > 1e-12)
    M = np.vstack([A3[T].T, b3[T][None, :]])                   # 101 x |T|
    if np.linalg.matrix_rank(M) == len(T): break               # columns independent: cannot shrink further
    z = np.linalg.svd(M)[2][-1]                                # a null vector: M z = 0
    if z.max() <= 0: z = -z
    pos = z > 1e-14
    theta = np.min(q[T][pos] / z[pos])                         # largest step keeping q >= 0
    q[T] = np.maximum(q[T] - theta * z, 0); q[q < 1e-12] = 0
    history.append(int(np.sum(q > 0)))
T = np.flatnonzero(q > 0)
print(f"    after {len(history) - 1} reduction steps: support {len(T)},  ||A'q|| = {np.linalg.norm(A3.T @ q):.1e},  b'q = {b3 @ q:.4f}")
print("    reduced subsystem infeasible (independent LP check):", infeasible(T))
np.savez("p3_data.npz", p=p, history=np.array(history))
