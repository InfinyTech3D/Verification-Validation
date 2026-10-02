# Architecture

This repository checks that SOFA's FEM components are correct. It tests components of SOFA core,
not code of its own. It does so in two independent ways:

- **Verification** checks that the code solves its equations right. The Method of Manufactured
  Solutions (MMS) is employed for this purpose. SOFA solves a problem whose exact solution is known,
  and the error must shrink at the rate the discretisation promises.
- **Validation** checks that it solves the right equations: SOFA and FEniCS solve the same
  physical problem, and their results must agree.

## Layers

Both suites are built from the same four layers. Each one calls the layer below it.

| Layer | Job | Verification | Validation |
|---|---|---|---|
| **Runner** | find cases, run a study on each, summarise, set the exit code | `verification/run.py` | `validation/run.py` |
| **Study** | the experiment: what to solve, what to measure, the verdict, the figures | `verification/study/study.py` | `validation/study/study.py` |
| **Scene** | one SOFA problem, built and ready to solve | `verification/scene/scene.py` | each case's `sofa_scene.py` |
| **Metric** | one number from one solve | `verification/study/metrics.py` | `validation/study/metrics.py` |

A **case** feeds the study its parameters and its pass criteria. In verification it is a JSON file
under `verification/cases/<ForceField>/`. In validation it is a folder under
`validation/cases/<Component>/<dim>D/<Case>/`.

The runner does not know how a study reaches its verdict, and a study does not know how many cases
there are. A new kind of experiment is therefore a new study; the runner, scenes and metrics are
reused as they are.

## Layout

```
common/                    shared by both suites
  geometry/                shapes and their named boundary regions (Bar1D, Beam2D, Tire3D, ...)
  sofa/                    scene base, prefabs, RegionClamp, SOFA naming conventions
  fenics/                  FEniCS meshes matching the SOFA ones
verification/
  run.py                   runner
  case.py                  loads and validates a case file; name -> class registries
  manufactured/            exact solutions, one module per geometry family
                           materials.py: constitutive laws, each a strain energy density
                           problem.py: ManufacturedProblem, a solution paired with a material
  scene/                   scene.py: MMSScene
                           controllers.py: body force, traction and load ramp
  study/                   study.py: ErrorConvergenceStudy, NormAgreementStudy
                           metrics.py: L2, H1, EnergyDifference, and the Measurement they read
                           quadrature.py: integration over the mesh, used by the metrics
  cases/<ForceField>/**.json   the cases
validation/
  run.py                   runner
  study/                   study.py: ISFComparisonStudy
                           metrics.py: RMSRelative
  cases/<Component>/<dim>D/<Case>/   the cases
```

## Runners

A runner takes one case or `--all`. For each case it builds the study, calls `run()` then
`write_results()`, and keeps the study. At the end it prints one `overview` line per case. A case
that raises does not stop the others. The verification runner also exits with an error when any
case failed or raised.

The verification runner chooses the study from the case: `NormAgreementStudy` when the case has
`compareAgainst`, `ErrorConvergenceStudy` otherwise. It also serves as a `runSofa` scene, to
inspect one mesh level of a case in the GUI:

```bash
runSofa -l SofaPython3 verification/run.py --argv <case.json> --argv <level>
```

Studies write to `verification/results/` and `validation/results/`, at the case's own path. Both
are ignored by git.

## Verification studies

`ErrorConvergenceStudy` solves one case on a sequence of refined meshes and judges the observed
order of accuracy:

```
run()
 ├─ for each mesh level:
 │    solve(cells)      build MMSScene, let SOFA solve
 │    measure(...)      Measurement ─► every metric ─► one value per metric
 │    rate(metric)      observed order over the two finest levels so far
 ├─ evaluate_metrics()  settled order vs expectedOrder ─► self.verified
 ├─ plot()              figures into self.plots
 └─ report()            the console table
write_results()         self.plots and record() into verification/results/
```

What goes into the scene is prepared before the study starts. `Case` loads the case and pairs the
manufactured solution with the material in a `ManufacturedProblem`. That object derives the stress
σ, the body force f = −∇·σ and the boundary traction from the exact displacement with sympy.
`MMSScene` then applies them through the controllers in `verification/scene/controllers.py`,
and clamps the Dirichlet regions with `RegionClamp` from `common/sofa/controllers.py`.

A `Measurement` holds SOFA's solution and the exact one at the quadrature points of the mesh.
Every metric reads from it, and the norms are integrated with SOFA's own shape functions.

`NormAgreementStudy` subclasses `ErrorConvergenceStudy`. It runs the same solves, then also
requires every level's errors to match those recorded by another case. For instance, a rotated
corotational problem must reproduce the linear one. It reads that record from
`verification/results/`, so the runner orders comparison cases after the cases they read.

## Validation study

`ISFComparisonStudy`, for inter-software comparison, runs one case folder:

| File | Provides |
|---|---|
| `case.json` | parameters and tolerances, either one block or one block per element type |
| `sofa_scene.py` | `solve(case)`, returning node positions and displacements |
| `fenics_scene.py` | `main(case)`, which writes `reference_solution.json` |
| `reference_solution.json` | the committed FEniCS result, so a run needs no FEniCS |
| `analytic_solution.py` (optional) | `displacement(x, case)`, an exact solution when one exists |

For each element type it solves the SOFA scene and checks that the meshes have the same nodes. It
then compares the displacements with every metric against `tolerance`, and against
`toleranceAnalytic` when an analytic solution exists. `--fenics` reruns `fenics_scene.py` to
regenerate the reference first.

## Writing a new study

**Same solves, different verdict.** Subclass `ErrorConvergenceStudy`, as `NormAgreementStudy`
does:
1. Override `evaluate_metrics`: call `super()`, then add your own verdict to `self.verified`.
2. Override `record`, `plot` and `report` to show it, each extending `super()`.

**A different experiment**, e.g. sweeping a load or a time step instead of the mesh: subclass
and override `run`. `MMSScene`, `measure` and the metrics are reused.

Then wire it into the runner:
1. Choose the study in `run()` in `verification/run.py`, from a key of the case.
2. Add that key to `OPTIONAL` in `verification/case.py`, and read it in `Case`.
3. `overview` reads `results`, `unconverged_levels` and, when present, `agreement`. Provide them,
   or give the new study its own line in `overview`.

A new validation study follows the same shape: `run()`, `write_results()`, and the attributes
`overview` in `validation/study/study.py` reads (`results`, `passed`, `max`, `analytic_results`,
`analytic_passed`).

## Extending the building blocks

**A verification case**
1. Add a JSON file under `verification/cases/<ForceField>/`. `LinearSmallStrainFEMForceField/Trigonometric2D_Quad.json`
   is a complete example, and `verification/case.py` lists every key.
2. Run it alone first. `--all` only visits the folders listed in `TESTED_COMPONENTS` in
   `verification/run.py`.

**A material**
1. Subclass `Material` in `verification/manufactured/materials.py`: `energy_density`,
   `get_material_parameters`, and `sofa_component_name` when SOFA takes the law as a separate
   component.
2. Add it to `MATERIALS` in `verification/case.py`.

**A manufactured solution**
1. Subclass `ManufacturedSolution` (or `TrigonometricSolution`) in the module of its geometry
   family under `verification/manufactured/`. Set `geometry` and `dim`, and write
   `displacement`.
2. Name it `<Type><dim>D` and export it from `manufactured/__init__.py`. A case refers to it as
   `<Type>`, and the dimension is taken from the geometry.

**A metric**
1. Subclass `Metric` in the suite's `metrics.py`.
2. Add an instance to `METRICS`. Every case must then state it: `expectedOrder` must name exactly
   the metrics in `METRICS` for a verification case, and `tolerance` must name it for a
   validation case.

**A geometry**
1. Subclass `Geometry` in `common/geometry/`: `dim`, `PARAMETER_ORDER` and the named boundary
   regions.
2. Add a prefab for it under `common/sofa/prefabs/`, and map the geometry to it in
   `PREFAB_BY_GEOMETRY`.
3. Add it to `GEOMETRIES` in `verification/case.py`.

**A validation case**
1. Create `validation/cases/<Component>/<dim>D/<Case>/` with the files above.
2. Run it with `--fenics` once to write the reference, and commit `reference_solution.json`.
