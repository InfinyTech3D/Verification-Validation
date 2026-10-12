"""Runner for the verification suite: loads a case and runs its verification study."""

import argparse
import json
import os
import pathlib
import sys
import traceback

# Make the plugin's packages importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verification.case import Case
from verification.scene.scene import MMSScene
from verification.study.study import (ErrorConvergenceStudy, NormAgreementStudy, clear_results,
                                overview)

CASES_ROOT = pathlib.Path(__file__).parent / "cases"


def resolve_path(path):
    """`path` as given when it is absolute or already exists, else taken from the cases folder."""
    path = pathlib.Path(path)
    return path if path.is_absolute() or path.exists() else CASES_ROOT / path


def run_all(directory, args):
    """Run every case under `directory`, each with its own table, then one line per case.

    Returns [(name, study)], the study being None where that case raised.
    """
    TESTED_COMPONENTS = ("LinearSmallStrainFEMForceField", "CorotationalFEMForceField",
                         "HyperelasticityFEMForceField")

    results = []
    for component in TESTED_COMPONENTS:
        # A comparison reads a record an earlier case wrote, so it goes last among its siblings.
        cases = sorted(sorted((directory / component).rglob("*.json")),
                       key=lambda path: "compareAgainst" in json.loads(path.read_text()))
        for path in cases:
            name = path.relative_to(directory).with_suffix("").as_posix()
            print(f"\n--- {name} ---")
            try:
                results.append((name, run(path)))
            except Exception as error:
                # One broken case should not stop the others.
                print(f"  failed: {type(error).__name__}: {error}")
                # Keep the traceback right below the case that raised it.
                if args.traceback:
                    traceback.print_exc(file=sys.stdout)
                results.append((name, None))
    overview(results)
    return results


def run(case_path):
    """Load one case, run its verification study, and write what it produced to disk."""
    case = Case.load(case_path)
    study = NormAgreementStudy(case) if case.compare_against else ErrorConvergenceStudy(case)
    study.run()
    study.write_results()
    return study


def createScene(root):
    """The runSofa entry point, allowing to inspect a single mesh level of a case in the GUI.

        runSofa -l SofaPython3 run.py --argv <case> --argv <level>

    `<level>` indexes the case's refinement sequence, so `0` is its coarsest mesh.
    """
    if len(sys.argv) != 3:
        sys.exit("run.py under runSofa expects: --argv <case> --argv <level>")
    case = Case.load(resolve_path(sys.argv[1]))
    cells = case.levels()[int(sys.argv[2])]

    # Add VisualStyle
    root.addObject('RequiredPlugin', name='visual', pluginName=["Sofa.Component.Visual"])
    root.addObject('VisualStyle', displayFlags='showForceFields showBehaviorModels')

    # Build the scene exercising the given mesh level with the MMS
    MMSScene(case, cells).build(root)


def parse_arguments():
    """Parse the command line into the whole run configuration.

    Returns
    -------
    argparse.Namespace
        case : str or None
            Path to the single case to run. ``None`` when ``--all`` was given instead. Taken
            relative to the cases folder by ``resolve_path`` when it does not resolve as typed.
        all : bool
            Run every case under the cases folder's force-field subdirectories. Mutually
            exclusive with ``case``.
        clean : bool
            Delete what previous runs wrote before running.
        traceback : bool
            Whether a case that raises under ``--all`` also prints its full stack trace. False by
            default.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    # Must choose either --all or pass the case path
    which_cases = parser.add_mutually_exclusive_group()
    which_cases.add_argument("case", nargs="?", help="path to a case")
    which_cases.add_argument("--all", action="store_true", help="every case under the cases folder")
    parser.add_argument("--clean", action="store_true",
                        help="delete the results of previous runs before running")
    parser.add_argument("--traceback", action="store_true",
                        help="full stack for a case that raises under --all")

    arguments = parser.parse_args()
    if not (arguments.case or arguments.all or arguments.clean):
        parser.error("one of case, --all or --clean is required")
    return arguments


if __name__ == "__main__":
    args = parse_arguments()

    if args.clean:
        clear_results()

    # Run all the cases in verification
    if args.all:
        failed = [name for name, study in run_all(CASES_ROOT, args)
                  if study is None or not study.verified]
    # Run the prescribed case
    elif args.case:
        study = run(resolve_path(args.case))
        failed = [] if study.verified else [study.name]
    # Nothing left to do: --clean was given on its own
    else:
        failed = []

    # On failure, exit with error and report the problematic cases
    if failed:
        sys.exit(f"\nfailed: {', '.join(failed)}")
