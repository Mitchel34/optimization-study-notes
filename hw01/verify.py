"""Numerical sanity checks for Problems 3 and 4 of HW1."""
import numpy as np
rng = np.random.default_rng(1)

# ---- Problem 3: {x : ||x-v1|| <= ||x-v2||} == {x : c^T x <= d} ----
n = 5
v1, v2 = rng.standard_normal(n), rng.standard_normal(n)
c, d = 2 * (v2 - v1), v2 @ v2 - v1 @ v1
X = rng.standard_normal((100_000, n)) * 3
closer = np.linalg.norm(X - v1, axis=1) <= np.linalg.norm(X - v2, axis=1)
halfsp = X @ c <= d
print("Problem 3: memberships agree on all samples:", bool((closer == halfsp).all()),
      f"({closer.sum()} of {len(X)} points closer to v1)")
m = (v1 + v2) / 2
print("           midpoint on boundary: c^T m - d =", round(float(c @ m - d), 12))

# ---- Problem 4: convexity of {x : ||Ax+b||_2 <= c^T x + d} ----
A, b = rng.standard_normal((4, 3)), rng.standard_normal(4)
cc, dd = rng.standard_normal(3), 2.0
inside = lambda x: np.linalg.norm(A @ x + b) <= cc @ x + dd
pts = rng.standard_normal((20_000, 3))
S = np.array([p for p in pts if inside(p)])
i, j = rng.integers(len(S), size=(2, 5000)); lam = rng.random((5000, 1))
Z = lam * S[i] + (1 - lam) * S[j]
viol = sum(not inside(z) for z in Z)
print(f"Problem 4: {len(S)} points in set; {viol} of 5000 random convex combinations left the set")
