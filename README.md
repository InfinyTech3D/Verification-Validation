# Verification-Validation
Initially funded by TIRREX, this repository proposes a framework for verification and validation. It relies on both manufactured solutions and comparison with alternative simulation technologies. This work therefore proves the reliability and compliance of the SOFA models &amp; algorithms.

For how the repository is organised, how the suite works and how to extend it, see
[docs/architecture.md](docs/architecture.md).

## Verification

**Verification** uses the method of manufactured solutions: an exact displacement is chosen, the
loads that produce it are derived, and SOFA solves the problem on a sequence of refined meshes.
The error must shrink at the rate the discretisation predicts. Covered force fields:
- `LinearSmallStrainFEMForceField`

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
python verification/run.py --all                # every test case
python verification/run.py --clean              # delete previous results
```

Results are written to `verification/results/`.
