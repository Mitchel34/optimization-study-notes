# Optimization Study Notes

Personal study guides, worked solutions, and numerical experiments for graduate study in optimization at The University of Texas at Austin. The notes connect geometric intuition, mathematical proofs, and executable examples to build the linear algebra and optimization background used in machine learning.

The broader course covers convex sets and functions, linear programming, convex programming, and iterative first- and second-order methods. Its organizing themes are **modeling**, **theory**, and **algorithms**: formulate a useful optimization problem, understand why a solution or certificate is valid, and choose a method for computing it. This repository currently covers the topics listed below; later algorithmic material will be added as it is studied.

These are personal companion notes, not official course materials or an instructor answer key. Lecture slides, assignment handouts, and other students' submissions are not included.

## Study materials

| Unit | Main connections | Read |
| --- | --- | --- |
| Week 1 | Convex sets and functions; hyperplanes and half-spaces; separation; epigraphs; quasiconvexity; proof techniques | [Study guide](hw01/hw01_study_guide.pdf) · [Worked solutions](hw01/hw01_solutions.pdf) |
| Week 3 | LP duality; complementary slackness; infeasibility certificates; Farkas' lemma; sparse certificates and Helly's theorem | [Study guide](hw03/hw03_study_guide.pdf) · [Worked solutions](hw03/hw03_solutions.pdf) · [Expanded solutions](hw03/hw03_solutions_expanded.pdf) |
| Week 4 | Stationary distributions via duality; piecewise linear minimization; optimality certificates; robust LPs and support functions | [Study guide](hw04/hw04_study_guide.pdf) · [Worked solutions](hw04/hw04_solutions.pdf) |

Each unit includes editable LaTeX source and Python examples. The study guides explain connected ideas and common mistakes; the solution documents develop the associated arguments. Numerical checks illustrate the theory and help detect implementation mistakes. They do not replace proofs.

## Suggested reading path

1. Start with Week 1 to connect algebraic inequalities with convex geometry.
2. Use Week 3 to learn how a dual feasible point certifies a bound, optimality, or infeasibility.
3. Read Week 4 as applications of that toolkit: balance affine pieces, prove an invariant distribution exists, and replace infinitely many uncertainty scenarios with finite constraints.
4. Reproduce the examples, then change a matrix, sign restriction, or uncertainty set and predict what should change before running the code.

## Run the numerical examples

Week 4's verification script uses only the Python 3 standard library and writes `hw04/verification_results.json`. The earlier scripts use NumPy, SciPy, and Matplotlib. A virtual environment keeps these dependencies local:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy matplotlib
```

Run each unit from its own directory because some scripts read and write relative paths:

```sh
cd hw01
python verify.py
python make_figures.py
cd ..

cd hw03
python verify.py
python make_figures.py
python focused_example.py
cd ..

cd hw04
python verify.py
cd ..
```

In Week 3, `verify.py` regenerates `p3_data.npz`, which `make_figures.py` reads. The scripts use fixed seeds where random examples are involved, but floating-point residuals and solver choices may vary across environments. Saved output files document example runs; rerun the scripts to check your environment.

## Edit and rebuild the documents

The PDF files are ready to read. To modify them, edit the corresponding `.tex` files. A LaTeX installation with `latexmk` and the packages named in each document's preamble can rebuild them from their unit directory:

```sh
cd hw01
latexmk -pdf hw01_study_guide.tex hw01_solutions.tex
cd ..

cd hw03
latexmk -pdf hw03_study_guide.tex hw03_solutions.tex hw03_solutions_expanded.tex
cd ..

cd hw04
latexmk -pdf hw04_study_guide.tex hw04_solutions.tex
cd ..
```

The earlier units use the checked-in figures and code listings. Regenerate the figures with the commands above when changing their examples.

## Repository scope

The repository contains personal notes, their rendered PDFs, illustrative figures, and supporting code and synthetic data. Its `.gitignore` uses an explicit allowlist so downloaded course documents, peer submissions, local metadata, environments, and build output stay outside version control. Add future material to that allowlist only after reviewing it for publication.
