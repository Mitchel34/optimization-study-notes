from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A = np.array([[0, 1], [1, -1], [-1, -1]])
b = np.array([0, -1, -1])
p = np.array([2, 1, 1])
assert (p >= 0).all() and (A.T @ p == 0).all() and b @ p < 0
# Witnesses after omitting rows 0, 1, and 2, respectively.
for omitted, x in enumerate([[0, 1], [1, 0], [-1, 0]]):
    keep = np.arange(3) != omitted
    assert (A[keep] @ x <= b[keep]).all()
print("Certificate: A.T @ p =", A.T @ p, "; b @ p =", b @ p)
print("Every pair of inequalities is feasible: True")

u = np.linspace(-2, 2, 401)
fig, ax = plt.subplots(figsize=(6.4, 2.5), layout="constrained")
ax.fill_between(u, -1.2, 0, color="#2463A6", alpha=0.2)
ax.fill_between(u, 1 + abs(u), 3.4, color="#CE7000", alpha=0.2)
ax.axhline(0, color="#2463A6", label=r"$x_2\leq 0$")
ax.plot(u, 1 + abs(u), color="#CE7000",
        label=r"$x_2\geq x_1+1$ and $x_2\geq -x_1+1$")
ax.set(xlim=(-2, 2), ylim=(-1.2, 3.4), xlabel=r"$x_1$",
       ylabel=r"$x_2$", xticks=[-2, -1, 0, 1, 2])
ax.legend(loc="upper center", fontsize=9, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
out = Path(__file__).resolve().parent / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "p3_focused_example.pdf")
plt.close(fig)
