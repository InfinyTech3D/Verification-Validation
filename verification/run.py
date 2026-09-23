"""Runner for the verification suite: loads a deck and runs its verification study."""

import argparse
import json
import os
import pathlib
import sys
import traceback

# Make the plugin's packages importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verification.deck import Deck
from verification.scene import MMSScene
from verification.study import (ErrorConvergenceStudy, NormAgreementStudy, clear_results,
                                overview)

# This file's own directory, the verification package. Decks live one directory down, grouped by
# the force field under test.
VERIFICATION_ROOT = pathlib.Path(__file__).parent


def resolve_path(path):
    """`path` as given when it is absolute or already exists, else taken from the verification root."""
    path = pathlib.Path(path)
    return path if path.is_absolute() or path.exists() else VERIFICATION_ROOT / path


def run_all(directory, args):
    """Run every deck under `directory`, each with its own table, then one line per deck.

    Returns [(name, study)], the study being None where that deck raised.
    """
    TESTED_COMPONENTS = ("LinearSmallStrainFEMForceField", "CorotationalFEMForceField")

    results = []
    for component in TESTED_COMPONENTS:
        # A comparison reads a record an earlier deck wrote, so it goes last among its siblings.
        decks = sorted(sorted((directory / component).glob("*.json")),
                       key=lambda path: "compareAgainst" in json.loads(path.read_text()))
        for path in decks:
            name = f"{path.parent.name}/{path.stem}"
            print(f"\n--- {name} ---")
            try:
                results.append((name, run(path)))
            except Exception as error:
                # One broken deck should not stop the others.
                print(f"  failed: {type(error).__name__}: {error}")
                # Keep the traceback right below the deck that raised it.
                if args.traceback:
                    traceback.print_exc(file=sys.stdout)
                results.append((name, None))
    overview(results)
    return results


def run(deck_path):
    """Load one deck, run its verification study, and write what it produced to disk."""
    deck = Deck.load(deck_path)
    study = NormAgreementStudy(deck) if deck.compare_against else ErrorConvergenceStudy(deck)
    study.run()
    study.write_results()
    return study


def createScene(root):
    """The runSofa entry point, allowing to inspect a single mesh level of a deck in the GUI.

        runSofa -l SofaPython3 run.py --argv <deck> --argv <level>

    `<level>` indexes the deck's refinement sequence, so `0` is its coarsest mesh.
    """
    if len(sys.argv) != 3:
        sys.exit("run.py under runSofa expects: --argv <deck> --argv <level>")
    deck = Deck.load(resolve_path(sys.argv[1]))
    cells = deck.levels()[int(sys.argv[2])]

    # Add VisualStyle
    root.addObject('RequiredPlugin', name='visual', pluginName=["Sofa.Component.Visual"])
    root.addObject('VisualStyle', displayFlags='showForceFields showBehaviorModels')

    # Build the scene exercising the given mesh level with the MMS
    MMSScene(deck, cells).build(root)


def parse_arguments():
    """Parse the command line into the whole run configuration.

    Returns
    -------
    argparse.Namespace
        deck : str or None
            Path to the single deck to run. ``None`` when ``--all`` was given instead. Taken
            relative to the verification root by ``resolve_path`` when it does not resolve as typed.
        all : bool
            Run every deck under the verification root's force-field subdirectories. Mutually
            exclusive with ``deck``.
        clean : bool
            Delete what previous runs wrote before running.
        traceback : bool
            Print the full stack for a deck that raises under ``--all``. False by default, where
            such a deck reports one line instead so the remaining decks still run.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    # Must choose either --all or pass the deck path
    which_decks = parser.add_mutually_exclusive_group()
    which_decks.add_argument("deck", nargs="?", help="path to a deck")
    which_decks.add_argument("--all", action="store_true", help="every deck under the verification root")
    parser.add_argument("--clean", action="store_true",
                        help="delete the results of previous runs before running")
    parser.add_argument("--traceback", action="store_true",
                        help="full stack for a deck that raises under --all, which otherwise "
                             "reports one line so the remaining decks still run")

    arguments = parser.parse_args()
    if not (arguments.deck or arguments.all or arguments.clean):
        parser.error("one of deck, --all or --clean is required")
    return arguments


if __name__ == "__main__":
    args = parse_arguments()

    if args.clean:
        clear_results()

    # Run all the decks in verification
    if args.all:
        failed = [name for name, study in run_all(VERIFICATION_ROOT, args)
                  if study is None or not study.verified]
    # Run the prescribed deck case
    elif args.deck:
        study = run(resolve_path(args.deck))
        failed = [] if study.verified else [study.name]
    # Nothing left to do: --clean was given on its own
    else:
        failed = []

    # On failure, exit with error and report the problematic decks
    if failed:
        sys.exit(f"\nfailed: {', '.join(failed)}")
