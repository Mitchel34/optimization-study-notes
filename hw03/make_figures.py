"""Figures for HW3 solutions and study guide (run verify.py first: it writes p3_data.npz)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREEN, RED, GRAY = "#1f5fa8", "#e07b1a", "#2a9d5c", "#c9302c", "#777777"
OUT = "figures/"
def save(fig, name): fig.tight_layout(); fig.savefig(OUT + name + ".pdf"); plt.close(fig)
def arrow(ax, p0, p1, color, lw=2): ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="->", color=color, lw=lw))

# ---------- P1: objective and constraint values along the unbounded ray ----------
c = np.array([1.0, -1, 0, 0]); A = np.array([[2.0, 3, -1, 1], [3, 1, 4, -2], [-1, -1, 2, 1]])
x0, d = np.array([0.0, 0, 2, 2]), np.array([-1.0, 0, 1, -3])
th = np.linspace(0, 5, 50); X = x0[:, None] + th * d[:, None]
fig, ax = plt.subplots(figsize=(5.6, 3.3))
ax.plot(th, c @ X, color=RED, lw=2.5, label=r"objective $x_1-x_2=-\theta$")
ax.plot(th, A[0] @ X, color=BLUE, label=r"row 1 $=-6\theta\ (\leq 0)$")
ax.plot(th, A[1] @ X, color=GREEN, label=r"row 2 $=4+7\theta\ (\geq 3)$")
ax.plot(th, A[2] @ X, color=ORANGE, label=r"row 3 $=6$")
ax.axhline(0, color=GRAY, lw=.6); ax.set_xlabel(r"$\theta$  along  $x^0+\theta d$"); ax.legend(fontsize=8.5, loc="upper left")
ax.set_title("Problem 1: feasible for every $\\theta\\geq 0$, cost $\\to-\\infty$")
save(fig, "p1_ray")

# ---------- P2: the two alternatives in R^2 ----------
def cone_panel(ax, M, title, cert=None):
    g = np.linspace(-.5, 3, 500); X1, X2 = np.meshgrid(g, g)
    inside = (X1 >= 0) & (X2 >= 0)
    for r in M: inside &= (r[0] * X1 + r[1] * X2 >= -1e-12)
    ax.contourf(X1, X2, inside.astype(float), levels=[.5, 1.5], colors=[GREEN], alpha=.35)
    for r, col, lab in zip(M, [BLUE, ORANGE], ["$a_1$", "$a_2$"]):
        t = np.array([-4, 4]); nrm = np.array([-r[1], r[0]])
        ax.plot(t * nrm[0], t * nrm[1], color=col, lw=1.2, ls="--")
        u = r / np.linalg.norm(r); arrow(ax, (0, 0), u, col); ax.annotate(lab, u * 1.15, color=col, ha="center", va="center")
    if cert is not None:
        v = M.T @ cert; arrow(ax, (0, 0), v, RED, 2.5); ax.annotate("$A^Tp$", v + np.array([-.1, -.38]), color=RED, ha="center")
    ax.axhline(0, color="k", lw=.8); ax.axvline(0, color="k", lw=.8)
    ax.set_xlim(-1.6, 3); ax.set_ylim(-1.6, 3); ax.set_aspect("equal"); ax.set_title(title, fontsize=10)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.7))
cone_panel(axs[0], np.array([[-2.0, 1], [1, -1]]), "(i) holds: cone is $\\{0\\}$; certificate $p=(1,1)$", cert=np.array([1.0, 1]))
axs[0].plot(0, 0, "o", color=GREEN, ms=7)
cone_panel(axs[1], np.array([[-1.0, 2], [1, -1]]), "(i) fails: cone has points with $x_1>0$; no $p$ exists")
save(fig, "p2_alternatives")

# ---------- P3: n = 2 picture ----------
H = np.array([[0.0, 1], [1, -1], [-1, -1], [1, 0], [-1, 0], [0, 1]]); h = np.array([0.0, -1, -1, 4, 4, 3.5])
g = np.linspace(-5, 5, 600); X1, X2 = np.meshgrid(g, g)
def boundary(ax, a, beta, col, lw, anchor):
    """Draw the line a'x = beta and a short arrow from `anchor` (slid along the line) into the side a'x <= beta."""
    a = np.asarray(a, float); tang = np.array([-a[1], a[0]]) / np.linalg.norm(a); base = a * beta / (a @ a)
    ax.plot(*np.array([base - 12 * tang, base + 12 * tang]).T, color=col, lw=lw)
    q = base + anchor * tang; arrow(ax, q, q - .9 * a / np.linalg.norm(a), col, 1.5)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.7))
for i, anc in zip(range(6), [3.5, 2.2, -2.2, 3.0, 3.0, -3.0]):
    boundary(axs[0], H[i], h[i], RED if i < 3 else GRAY, 2.2 if i < 3 else 1.2, anc)
axs[0].set_title("6 inequalities in $\\mathbb{R}^2$ (arrows: allowed side)", fontsize=10)
axs[1].contourf(X1, X2, (X2 <= 0).astype(float), levels=[.5, 1.5], colors=[BLUE], alpha=.3)
axs[1].contourf(X1, X2, (X2 >= 1 + np.abs(X1)).astype(float), levels=[.5, 1.5], colors=[ORANGE], alpha=.4)
for i, anc in zip(range(3), [3.5, 2.2, -2.2]): boundary(axs[1], H[i], h[i], RED, 2.2, anc)
axs[1].annotate("$x_2\\leq 0$", (-0.9, -3.6), color=BLUE); axs[1].annotate("$x_2\\geq 1+|x_1|$", (-1.75, 4.0), color=ORANGE)
axs[1].set_title("the 3 red ones alone are infeasible ($n+1=3$)", fontsize=10)
for ax in axs: ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
save(fig, "p3_helly2d")

# ---------- P3: dual vertex and support reduction ----------
dat = np.load("p3_data.npz"); p, hist = dat["p"], dat["history"]
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.2))
nz = np.flatnonzero(p > 1e-9)
axs[0].vlines(nz, 0, p[nz], color=BLUE, lw=1); axs[0].plot(nz, p[nz], ".", color=BLUE, ms=3)
axs[0].set_xlabel("inequality index $i$"); axs[0].set_ylabel("$p_i^\\star$"); axs[0].set_xlim(0, 500)
axs[0].set_title(f"Dual vertex: {len(nz)} of 500 multipliers nonzero", fontsize=10)
axs[1].plot(hist, color=ORANGE, lw=2); axs[1].axhline(101, color=RED, ls="--", lw=1); axs[1].annotate("$n+1=101$", (2, 104), color=RED, fontsize=9)
axs[1].set_xlabel("reduction step"); axs[1].set_ylabel("support of certificate")
axs[1].set_title("Shrinking a dense Farkas certificate", fontsize=10)
save(fig, "p3_support")

# ---------- Guide: weak/strong duality number line ----------
fig, ax = plt.subplots(figsize=(6.4, 1.9))
ax.axhline(0, color="k", lw=1)
ax.plot([3.2, 4, 5.3, 6.1], [0] * 4, "v", color=BLUE, ms=9); ax.plot([-.5, .8, 1.7, 2.4], [0] * 4, "^", color=ORANGE, ms=9)
ax.plot(2.8, 0, "*", color=RED, ms=16)
ax.annotate("primal feasible costs $c^Tx$", (4.7, .35), color=BLUE, ha="center"); ax.annotate("dual feasible values $p^Tb$", (.9, .35), color=ORANGE, ha="center")
ax.annotate("common optimum", (2.8, -.62), color=RED, ha="center", fontsize=9)
ax.set_ylim(-1, 1); ax.set_xlim(-1.2, 6.8); ax.axis("off")
save(fig, "g_weak_duality")

# ---------- Guide: Farkas geometry ----------
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4))
A1, A2 = np.array([2.0, .6]), np.array([.7, 2.0])
for ax, bvec, ttl in [(axs[0], np.array([1.6, 1.5]), "$b\\in$ cone: $Ax=b,\\ x\\geq 0$ solvable"), (axs[1], np.array([2.2, -.9]), "$b\\notin$ cone: separating $p$ exists")]:
    ax.add_patch(Polygon([(0, 0), 3 * A1, 3 * A2], color=GREEN, alpha=.25))
    arrow(ax, (0, 0), A1, GREEN); arrow(ax, (0, 0), A2, GREEN); ax.annotate("$A_1$", A1 + (.05, -.3), color=GREEN); ax.annotate("$A_2$", A2 + (-.45, 0), color=GREEN)
    arrow(ax, (0, 0), bvec, BLUE, 2.5); ax.annotate("$b$", bvec + (.08, .05), color=BLUE)
    ax.set_xlim(-1.5, 3.2); ax.set_ylim(-1.6, 3.2); ax.set_aspect("equal"); ax.axis("off"); ax.set_title(ttl, fontsize=10)
pv = np.array([-.6, 2.0]) / np.linalg.norm([-.6, 2.0]); t = np.array([-1.4, 3.2]); tang = np.array([pv[1], -pv[0]])
axs[1].plot(t * tang[0], t * tang[1], color=RED, ls="--"); arrow(axs[1], (0, 0), pv * 1.1, RED); axs[1].annotate("$p$", pv * 1.25, color=RED)
axs[1].annotate("$p^TA\\geq 0$ side", (-1.5, 1.75), color=RED, fontsize=8.5); axs[1].annotate("$p^Tb<0$", (1.6, -1.4), color=RED, fontsize=8.5)
save(fig, "g_farkas")

# ---------- Guide: complementary slackness / optimality geometry ----------
fig, ax = plt.subplots(figsize=(4.8, 3.9))
V = np.array([[0, 0], [3, 0], [3.6, 1.6], [2, 3], [0, 2.4]]); ax.add_patch(Polygon(V, color=BLUE, alpha=.2, ec=BLUE))
xs = V[0]
n1, n2, cvec = np.array([0, 1.0]), np.array([1.0, 0]), np.array([.55, .9])
arrow(ax, xs, xs + n1 * 1.2, GREEN); arrow(ax, xs, xs + n2 * 1.2, GREEN); arrow(ax, xs, xs + cvec * 1.25, RED, 2.5)
ax.annotate("$a_1$", xs + n1 * 1.3 + (-.3, 0), color=GREEN); ax.annotate("$a_2$", xs + n2 * 1.3 + (0, -.3), color=GREEN); ax.annotate("$c=p_1a_1+p_2a_2$", xs + cvec * 1.3 + (.05, .05), color=RED)
ax.plot(*xs, "o", color="k"); ax.annotate("$x^\\star$", xs + (-.35, -.3))
ax.annotate("inactive constraints:\n$p_i=0$", (2.5, 2.75), fontsize=8.5, color=GRAY)
ax.set_xlim(-.8, 4.4); ax.set_ylim(-.7, 3.6); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("min $c^Tx$ s.t. $a_i^Tx\\geq b_i$: $c$ lies in the cone\nof the active constraint normals", fontsize=10)
save(fig, "g_comp_slack")

# ---------- Guide: tight example for n+1 ----------
fig, ax = plt.subplots(figsize=(4.2, 3.6))
T = np.array([[0.0, 1], [1, -1], [-1, -1]]); tb = np.array([0.0, -1, -1])
for i, col in enumerate([BLUE, ORANGE, GREEN]):
    ax.contourf(X1, X2, (T[i, 0] * X1 + T[i, 1] * X2 <= tb[i]).astype(float), levels=[.5, 1.5], colors=[col], alpha=.22)
    ax.contour(X1, X2, T[i, 0] * X1 + T[i, 1] * X2 - tb[i], levels=[0], colors=[col], linewidths=1.5)
ax.set_xlim(-5, 5); ax.set_ylim(-4, 5); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Any 2 overlap, all 3 do not:\n$n+1$ cannot be improved", fontsize=10)
save(fig, "g_tight")
print("figures:", sorted(__import__("os").listdir("figures")))
