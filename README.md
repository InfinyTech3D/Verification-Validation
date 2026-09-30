# Verification-Validation
Initially funded by TIRREX, this repository proposes a framework for verification and validation. It relies on both manufactured solutions and comparison with alternative simulation technologies. This work therefore proves the reliability and compliance of the SOFA models &amp; algorithms.

For how the repository is organised, how both suites work and how to extend them, see
[docs/architecture.md](docs/architecture.md).

## Verification and validation

**Verification** uses the method of manufactured solutions: an exact displacement is chosen, the
loads that produce it are derived, and SOFA solves the problem on a sequence of refined meshes.
The error must shrink at the rate the discretisation predicts. Covered force fields:
- `LinearSmallStrainFEMForceField`
- `CorotationalFEMForceField`
- `HyperelasticityFEMForceField` covering the following material laws:
  - `Saint Venant-Kirchhoff`
  - `Neo-Hookean`
  - `Mooney-Rivlin`
  - `Ogden`

**Validation** compares SOFA against reference results computed with FEniCS, and against an
analytic solution where one exists. Covered force fields: 
- `LinearSmallStrainFEMForceField`

## Layout

```
common/                                  code shared by both suites: geometry, SOFA and FEniCS helpers
verification/                            the verification suite
  cases/<Component>/**.json              one test case per file
validation/                              the validation suite
  cases/<Component>/<dim>D/<Case>/       sofa_scene.py, fenics_scene.py, case.json,
                                         reference_solution.json, analytic_solution.py (optional)
```

## Requirements

- SOFA with SofaPython3
- numpy, sympy, matplotlib
- FEniCSx (dolfinx, petsc4py, mpi4py), only to regenerate validation references with `--fenics`

## Running

```bash
python verification/run.py <path-to-json-file>  # one test case
python verification/run.py --all                # every test case
python verification/run.py --clean              # delete previous results
```

```bash
python validation/run.py <path-to-case-folder>  # one case
python validation/run.py --all                  # every case
python validation/run.py --all --fenics         # regenerate the FEniCS references first
```

Results are written to `verification/results/` and `validation/results/`.
