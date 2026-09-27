"""Generate all figures for HW1 solutions and the study guide."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREEN, RED, GRAY = "#1f5fa8", "#e07b1a", "#2a9d5c", "#c9302c", "#777777"
OUT = "figures/"

# ---------- Fig 1: union of two convex sets is not convex ----------
fig, ax = plt.subplots(figsize=(5, 3.2))
ax.add_patch(Circle((-1.5, 0), 1, color=BLUE, alpha=.35, label="$C_1$"))
ax.add_patch(Circle((1.5, 0), 1, color=ORANGE, alpha=.35, label="$C_2$"))
x, y = np.array([-1.5, 0]), np.array([1.5, 0])
ax.plot([x[0], y[0]], [x[1], y[1]], "--", color=GRAY)
ax.plot(*x, "o", color=BLUE); ax.plot(*y, "o", color=ORANGE); ax.plot(0, 0, "x", color=RED, ms=10, mew=2)
ax.annotate("$x$", x + (-.05, .15)); ax.annotate("$y$", y + (-.05, .15))
ax.annotate(r"$\frac{1}{2} x+\frac{1}{2} y \notin C_1\cup C_2$", (0, -.35), ha="center", color=RED)
ax.set_xlim(-3, 3); ax.set_ylim(-1.5, 1.5); ax.set_aspect("equal"); ax.legend(loc="upper right")
ax.set_title("Union of two convex sets need not be convex")
fig.tight_layout(); fig.savefig(OUT + "union_not_convex.png", dpi=200); plt.close(fig)

# ---------- Fig 2: min of two convex functions is not convex ----------
t = np.linspace(-1.5, 3.5, 400)
f1, f2 = t**2, (t - 2) ** 2
fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.plot(t, f1, color=BLUE, lw=1, alpha=.6, label="$f_1(x)=x^2$")
ax.plot(t, f2, color=ORANGE, lw=1, alpha=.6, label="$f_2(x)=(x-2)^2$")
ax.plot(t, np.minimum(f1, f2), color=RED, lw=2.5, label=r"$f_{\min}=\min\{f_1,f_2\}$")
ax.plot([0, 2], [0, 0], "o--", color=GRAY)
ax.plot(1, 1, "s", color=RED); ax.annotate(r"$f_{\min}(1)=1 > 0=\frac{1}{2} f_{\min}(0)+\frac{1}{2} f_{\min}(2)$", (1.1, 1.05), fontsize=9)
ax.set_ylim(-.3, 3); ax.set_xlim(-1.3, 3.3); ax.legend(loc="upper center", fontsize=9)
ax.set_title("Pointwise minimum destroys convexity")
fig.tight_layout(); fig.savefig(OUT + "min_not_convex.png", dpi=200); plt.close(fig)

# ---------- Fig 2b: max of convex is convex (study guide) ----------
fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.plot(t, f1, color=BLUE, lw=1, alpha=.6, label="$f_1$"); ax.plot(t, f2, color=ORANGE, lw=1, alpha=.6, label="$f_2$")
ax.plot(t, np.maximum(f1, f2), color=GREEN, lw=2.5, label=r"$f_{\max}=\max\{f_1,f_2\}$")
ax.set_ylim(-.3, 6); ax.set_xlim(-1.3, 3.3); ax.legend(loc="upper center", fontsize=9)
ax.set_title("Pointwise maximum preserves convexity")
fig.tight_layout(); fig.savefig(OUT + "max_convex.png", dpi=200); plt.close(fig)

# ---------- Fig 3: disjoint closed convex sets, no strict separation ----------
fig, ax = plt.subplots(figsize=(5.2, 3.4))
xs = np.linspace(0.12, 6, 400)
ax.fill_between(xs, 1 / xs, 9, color=ORANGE, alpha=.35, label=r"$C_2=\{(x,y):x>0,\ y\geq 1/x\}$")
ax.fill_between([-1, 6], [0, 0], [-3, -3], color=BLUE, alpha=.35, label=r"$C_1=\{(x,y):y\leq 0\}$")
ax.axhline(0, color=GRAY, ls="--", lw=1.2, label="only separating line: $y=0$")
ax.set_xlim(-1, 6); ax.set_ylim(-1.5, 4); ax.legend(loc="upper right", fontsize=8.5)
ax.set_title("Disjoint, closed, convex, not strictly separable")
fig.tight_layout(); fig.savefig(OUT + "no_strict_separation.png", dpi=200); plt.close(fig)

# ---------- Fig 4: quasiconvex but not convex ----------
xs = np.linspace(-4, 4, 400)
g = np.sqrt(np.abs(xs))
fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.plot(xs, g, color=BLUE, lw=2.5, label=r"$f(x)=\sqrt{|x|}$")
c = 1.2
ax.axhline(c, color=GRAY, ls="--", lw=1); ax.annotate("$c$", (-3.9, c + .07))
ax.plot([-c**2, c**2], [-.08, -.08], lw=6, color=GREEN, solid_capstyle="butt", label=r"$L_c=[-c^2,c^2]$ (an interval $\Rightarrow$ convex)")
ax.plot([-3, 3], [np.sqrt(3), np.sqrt(3)], ":", color=RED, lw=1.5, label="chord lies below graph $\Rightarrow$ not convex")
ax.set_ylim(-.3, 2.3); ax.legend(loc="upper center", fontsize=8.5)
ax.set_title("Convex sublevel sets, non-convex function")
fig.tight_layout(); fig.savefig(OUT + "quasiconvex.png", dpi=200); plt.close(fig)

# ---------- Fig 5: Voronoi half-space for v1=(-1,0), v2=(1,0) ----------
fig, ax = plt.subplots(figsize=(5, 4))
ax.fill_between([-3, 0], [-3, -3], [3, 3], color=BLUE, alpha=.3, label=r"$\{x: \|x-v_1\|\leq\|x-v_2\|\}=\{x_1\leq 0\}$")
ax.axvline(0, color=BLUE, lw=2)
ax.plot(-1, 0, "o", color=BLUE, ms=8); ax.plot(1, 0, "o", color=ORANGE, ms=8)
ax.annotate("$v_1=(-1,0)$", (-1, .2), ha="center"); ax.annotate("$v_2=(1,0)$", (1, .2), ha="center")
ax.annotate("", xy=(1.2, -1.5), xytext=(0, -1.5), arrowprops=dict(arrowstyle="->", color=RED, lw=2))
ax.annotate("$c=(1,0)$", (0.6, -1.85), color=RED, ha="center")
ax.plot(-1.5, 1.5, "kx"); ax.annotate("$x$", (-1.45, 1.6))
ax.plot([-1.5, -1], [1.5, 0], "-", color=BLUE, lw=1); ax.plot([-1.5, 1], [1.5, 0], "-", color=ORANGE, lw=1)
ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal"); ax.axhline(0, color=GRAY, lw=.5)
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.legend(loc="lower left", fontsize=8.5)
ax.set_title("Points closer to $v_1$ than $v_2$: a half-space")
fig.tight_layout(); fig.savefig(OUT + "halfspace_voronoi.png", dpi=200); plt.close(fig)

# ---------- Fig 6 (study guide): convex vs nonconvex set ----------
fig, axs = plt.subplots(1, 2, figsize=(6.5, 3))
th = np.linspace(0, 2 * np.pi, 200)
axs[0].fill(1.4 * np.cos(th), np.sin(th), color=BLUE, alpha=.35)
axs[0].plot([-1, 1], [-.5, .6], "k-o", ms=4); axs[0].set_title("Convex: every chord stays inside")
star = np.array([[0, 1], [.3, .3], [1, .3], [.45, -.15], [.6, -1], [0, -.45], [-.6, -1], [-.45, -.15], [-1, .3], [-.3, .3]])
axs[1].add_patch(Polygon(star, color=ORANGE, alpha=.4)); axs[1].plot([-.9, .9], [.25, .25], "k-o", ms=4)
axs[1].set_title("Not convex: chord leaves the set")
for a in axs: a.set_xlim(-1.6, 1.6); a.set_ylim(-1.3, 1.3); a.set_aspect("equal"); a.axis("off")
fig.tight_layout(); fig.savefig(OUT + "convex_vs_nonconvex.png", dpi=200); plt.close(fig)

# ---------- Fig 7 (study guide): convex function chord + tangent ----------
xs = np.linspace(-2.2, 2.2, 300); f = xs**2 + .3 * xs
fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.plot(xs, f, color=BLUE, lw=2.5, label="$f$ (convex)")
a, b = -1.5, 1.5; fa, fb = a**2 + .3 * a, b**2 + .3 * b
ax.plot([a, b], [fa, fb], "o-", color=RED, label="chord: above graph")
x0 = .6; f0 = x0**2 + .3 * x0; s = 2 * x0 + .3
ax.plot(xs, f0 + s * (xs - x0), "--", color=GREEN, label="tangent: below graph (1st-order cond.)")
ax.set_ylim(-1, 5); ax.legend(fontsize=8.5, loc="upper center"); ax.set_title("Two pictures of convexity")
fig.tight_layout(); fig.savefig(OUT + "convex_function.png", dpi=200); plt.close(fig)

# ---------- Fig 8 (study guide): separating & supporting hyperplanes ----------
fig, ax = plt.subplots(figsize=(5, 3.6))
ax.add_patch(Circle((-1.3, 0), 1, color=BLUE, alpha=.35)); ax.add_patch(Circle((1.6, .3), .8, color=ORANGE, alpha=.35))
ax.plot([.25, .25], [-2, 2], color=GRAY, lw=2); ax.annotate("$a^Tx=b$", (.35, 1.6), color=GRAY)
ax.annotate("", xy=(1.0, -1.5), xytext=(.25, -1.5), arrowprops=dict(arrowstyle="->", color=RED, lw=2)); ax.annotate("$a$", (.6, -1.35), color=RED)
ax.annotate("$C$", (-1.3, 0), ha="center"); ax.annotate("$D$", (1.6, .3), ha="center")
ax.set_xlim(-3, 3); ax.set_ylim(-2, 2); ax.set_aspect("equal"); ax.axis("off"); ax.set_title("Separating hyperplane: $a^Tx\\leq b$ on $C$, $a^Tx\\geq b$ on $D$")
fig.tight_layout(); fig.savefig(OUT + "separating_hyperplane.png", dpi=200); plt.close(fig)

# ---------- Fig 9 (study guide): parallel hyperplane distance ----------
fig, ax = plt.subplots(figsize=(4.6, 3.6))
a = np.array([1, 1]) / np.sqrt(2); b1, b2 = 0.0, 1.5
for b, col in [(b1, BLUE), (b2, ORANGE)]:
    xs = np.linspace(-3, 3, 10); ax.plot(xs, (b - a[0] * xs) / a[1], color=col, lw=2)
p1 = a * b1; p2 = a * b2
ax.annotate("", xy=p2, xytext=p1, arrowprops=dict(arrowstyle="<->", color=RED, lw=2))
ax.annotate(r"$\frac{|b_1-b_2|}{\|a\|_2}$", (p1 + p2) / 2 + (.25, -.2), color=RED, fontsize=12)
ax.annotate("$a^Tx=b_1$", (-2.6, 2.1), color=BLUE); ax.annotate("$a^Tx=b_2$", (-1.2, 2.4), color=ORANGE)
ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal"); ax.axis("off"); ax.set_title("Distance between parallel hyperplanes")
fig.tight_layout(); fig.savefig(OUT + "parallel_hyperplanes.png", dpi=200); plt.close(fig)

# ---------- Fig 10: numerical check for Problem 4 (norm-cone type set) ----------
rng = np.random.default_rng(0)
A = np.array([[1.0, 0.4], [-0.3, 1.1]]); bb = np.array([0.2, -0.1]); cc = np.array([0.9, 0.6]); dd = 1.5
X1, X2 = np.meshgrid(np.linspace(-3, 3, 400), np.linspace(-3, 3, 400))
P = np.stack([X1, X2], -1)
lhs = np.linalg.norm(P @ A.T + bb, axis=-1); rhs = P @ cc + dd
fig, ax = plt.subplots(figsize=(4.6, 4))
ax.contourf(X1, X2, (lhs <= rhs).astype(float), levels=[.5, 1.5], colors=[GREEN], alpha=.4)
ax.contour(X1, X2, lhs - rhs, levels=[0], colors=[GREEN])
ax.set_aspect("equal"); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
ax.set_title(r"$\{x:\|Ax+b\|_2\leq c^Tx+d\}$ (Problem 4)")
fig.tight_layout(); fig.savefig(OUT + "problem4_set.png", dpi=200); plt.close(fig)
print("figures written:", sorted(__import__("os").listdir("figures")))
