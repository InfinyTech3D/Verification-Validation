"""Runner for the validation suite: loads a case and runs its ISFComparisonStudy."""

import argparse
import os
import pathlib
import sys
import traceback

# Make the plugin's packages importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation.study import ISFComparisonStudy, overview


def run_all(directory, args):
    """Run every case under `directory`, each with its own output, then one line per case.

    Returns [(name, study)], the study being None where that case raised.
    """
    results = []
    for path in sorted(directory.glob("*/*/*/sofa_scene.py")):
        case_dir = path.parent
        name = str(case_dir.relative_to(directory))
        print(f"\n--- {name} ---")
        try:
            results.append((name, run(case_dir, fenics=args.fenics)))
        except Exception as error:
            # One broken case should not stop the others.
            print(f"  failed: {type(error).__name__}: {error}")
            if args.traceback:
                traceback.print_exc(file=sys.stdout)
            results.append((name, None))
    overview(results)
    return results


def run(case_dir, fenics=False):
    """Run one case's comparison study, write the results it produced, and return it."""
    study = ISFComparisonStudy(case_dir)
    study.run(regenerate_fenics=fenics)
    study.write_results()
    return study


def parse_arguments():
    """Parse the command line into the whole run configuration.

    Returns
    -------
    argparse.Namespace
        case : str or None
            Path to the single case to run. ``None`` when ``--all`` was given instead.
        all : bool
            Run every case under the validation root. Mutually exclusive with ``case``, and
            exactly one of the two is always set.
        fenics : bool
            Regenerate each case's reference_solution.json by running its fenics_scene.py before
            comparing, instead of comparing against the already-committed reference. False by
            default.
        traceback : bool
            Whether a case that raises under ``--all`` also prints its full stack trace. False by
            default.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    # Must choose either --all or pass the case path
    which_cases = parser.add_mutually_exclusive_group(required=True)
    which_cases.add_argument("case", nargs="?", help="path to a case")
    which_cases.add_argument("--all", action="store_true", help="every case under the validation root")
    parser.add_argument("--fenics", action="store_true",
                        help="regenerate reference_solution.json by running fenics_scene.py before comparing")
    parser.add_argument("--traceback", action="store_true",
                        help="full stack for a case that raises under --all ")
    return parser.parse_args()


VALIDATION_ROOT = pathlib.Path(__file__).parent


if __name__ == '__main__':
    args = parse_arguments()
    if args.all:
        run_all(VALIDATION_ROOT, args)
    else:
        run(args.case, fenics=args.fenics)
