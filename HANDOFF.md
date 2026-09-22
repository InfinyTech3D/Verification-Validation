# Context handoff — validation suite, branch `dev-validation`

Temporary. Delete once the PETSc verification below is done and the next cases
are under way.

## Repo

Plugin repo `Verification-Validation`, worktree checked out as
`plugins/VnV-dev-validation` on branch `dev-validation`. On Windows it was under
`~/software/apps/SOFA/dev/worktrees/dev-vnv/plugins/VnV-dev-validation` — find the
equivalent path on this machine, do not assume it.

The repo is a V&V suite for SOFA's FEM framework, being cleanly transferred out of
the `Elasticity` plugin. Two halves: verification (manufactured solutions, MMS)
and validation (cross-checks against another FEM code). The verification half is
already ported. The validation half is what we are building; before this session
it did not exist at all.

## What was decided

- Validation uses **FEniCS (dolfinx)**, not FreeFem++. The original intern work
  used FreeFem; we are not porting those scripts, we are rewriting in FEniCS.
- Two validation methods: inter-software comparison against FEniCS (always), and
  an analytical solution (extra, when one exists).
- Layout: `validation/<ForceFieldName>/<dim>/<CaseName>/{sofa_scene,fenics_scene}.py`
- The directory is named `LinearSmallStrainFEMForceField` — that is the actual
  registered SOFA component. `LinearElasticityFEMForceField` does not exist; only
  the `BaseLinearElasticityFEMForceField` base class.
- Do **not** reuse the verification plumbing (`MMSScene`, `deck.py`, `study.py`,
  `registry.py`). Validation is deliberately less formulaic. **Do** reuse
  `common/geometry` and `common/sofa/prefabs`.
- No JSON parameter decks for validation. Config dicts are inlined in the scene.
- SOFA prefabs mesh only; they impose no boundary conditions. Each case places its
  own BCs, via `geometry.region_indices('left'/'right', nodes)` rather than
  hardcoded indices.
- Work one case at a time, complete, before starting the next. Modularize from what
  we learn, not ahead of it.

## What exists (one case, done and verified)

```
validation/LinearSmallStrainFEMForceField/1D/TractionBar/
    case.py          shared physical params + analytic solution
    sofa_scene.py    createScene() + canonical python entry point
    fenics_scene.py  dolfinx solve, writes fenics.json
    fenics.json      committed reference (CI has no FEniCS)
```

Case: bar on [0, 1], unit cross-section, clamped at x=0, axial point load F=1000
at x=1. E=1e6, ν=0.3, 9 P1 elements. Analytic u(x) = F·x/(E·A).

Both entry points verified working on Windows:

```
python sofa_scene.py                  # solves, compares, plots, prints max diff
runSofa -lSofaPython3 sofa_scene.py   # loads and steps clean
```

Agreement SOFA vs FEniCS: 1.08e-13 relative to max|u|. Plot lands in `results/`
(gitignored): unfilled blue squares = FEniCS, red crosshairs = SOFA, dashed line =
analytic.

## Physics finding that generalizes to every later case

SOFA's `toLameParameters<dim>` (`.../fem/elastic/impl/LameParameters.h`) is

```
mu     = E / (2(1+nu))
lambda = E*nu / ((1+nu)(1 - (dim-1)*nu))
```

So in 1D, λ + 2μ = E exactly: axial stiffness carries no Poisson effect and
`poissonRatio` is inert. **dim=2 is therefore plane stress**, dim=3 is standard.
The FEniCS side must apply the same reduction or the two codes are not comparable.
`fenics_scene.py` has this as `lame_parameters(dim)`.

## Why this moved to Linux — your main job

Windows has dolfinx 0.10.0 but **no petsc4py**, so `dolfinx.fem.petsc.LinearProblem`
is unusable there. `fenics_scene.py` calls `solve_with_petsc()` first and falls back
to `solve_with_scipy()` on `ImportError`. **The PETSc branch has never executed.**
On Linux petsc4py should be present. Please:

1. Report petsc4py and dolfinx versions.
2. Confirm the PETSc branch actually runs rather than assuming it (instrument it,
   then restore the file).
3. Check the `LinearProblem` call against the installed dolfinx. Two things are
   unverified: whether `petsc_options_prefix` is a real keyword, and whether
   `.solve()` returns the Function alone or a `(uh, converged_reason, iterations)`
   tuple. There is an `isinstance(solution, tuple)` guard for the second. Fix what
   is wrong and say what changed.
4. Regenerate `fenics.json` and diff against the committed scipy-path version.
   Expect round-off, ~1e-16 relative to max|u| = 1e-3. Anything larger is a real
   finding — report it, do not explain it away.
5. Only commit the regenerated `fenics.json` if the diff is round-off only.

Keep the scipy fallback. It costs ten lines and keeps small cases runnable on
Windows, which is where editing happens.

## Known outstanding issue

`solve_with_scipy` is correct **only on one MPI rank**: `A.to_scipy()` returns just
the local rows. `solve()` builds on `MPI.COMM_WORLD` with no rank guard, so
`mpirun -n 4 python fenics_scene.py` would silently write a wrong `fenics.json`.
Add a guard that raises rather than returning a wrong answer. Not yet done.

## Next cases, in order

1D distributed axial load, then the 2D cantilevers (distributed transverse and
axial compression, each in plane stress and plane strain; three-point bending;
self-weight), then 3D (distributed surface load, three-point bending, circular
section under traction and under bending, self-weight).

Sources for the physics are in `plugins/Elasticity/examples/Freefem/` — read them
for parameters and BCs only, do not port the code. Their per-case
`comparaison_script*.py` and `params*.json` are duplicated across cases and are
explicitly not being carried over.

## Environment notes

- SOFA env comes from `use-sofa <version>`; read `$SOFA_SRC` / `$SOFA_BUILD`, never
  hardcode. On Linux `spdlog` must stay disabled in the CMake preset.
- Python is conda env `py312`.
- Plugins build in-tree with SOFA: `sofa-build`, no separate plugin build step.
- Do not run the full VnV suite unless asked — runs are slow. The single 1D case
  above takes about a second and is fine to run.
