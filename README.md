# Verification-Validation
Initially funded by TIRREX, this repository proposes a framework for verification and validation. It relies on both manufactured solutions and comparison with alternative simulation technologies. This work therefore proves the reliability and compliance of the SOFA models &amp; algorithms.

## Verification

**Verification** uses the method of manufactured solutions: an exact displacement is chosen, the
loads that produce it are derived, and SOFA solves the problem on a sequence of refined meshes.
The error must shrink at the rate the discretisation predicts. Covered force fields:
- `LinearSmallStrainFEMForceField`

To get started, read [`verification/docs/IntroductionMMS.ipynb`](verification/docs/IntroductionMMS.ipynb): it presents the method and builds one verification study step by step, from the manufactured solution to the input file the suite runs.

## Layout

```
common/                                  shared code: geometry and SOFA helpers
verification/                            the verification suite
  cases/<Component>/**.json              one test case per file
```

## Requirements

- SOFA with SofaPython3
- numpy, sympy, matplotlib

## Running

```bash
python verification/run.py <path-to-json-file>  # one test case
python verification/run.py --all                # every test case, then one summary line per case
python verification/run.py --clean              # delete previous results
```

Results are written to `verification/results/`.

To inspect one mesh of a case in the SOFA GUI, the runner also acts as a SOFA scene file:

```bash
runSofa -l SofaPython3 verification/run.py --argv <path-to-json-file> --argv 0
```

The first argument is the path to the .json case file and the second is the mesh refinement level to be built.
