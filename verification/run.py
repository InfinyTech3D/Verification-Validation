"""Runner for the verification suite: loads a deck and runs its verification study."""

import argparse
import os
import pathlib
import sys
import traceback

import matplotlib.pyplot as plt

# Make the plugin's packages importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verification.deck import Deck
from verification.scene import MMSScene
from verification.study import ErrorConvergenceStudy, overview

# This file's own directory, the verification package. Decks live one directory down, grouped by
# the force field under test.
VERIFICATION_ROOT = pathlib.Path(__file__).parent
RESULTS_ROOT = VERIFICATION_ROOT / "results"


def resolve_path(path):
    """`path` as given when it is absolute or already exists, else taken from the verification root."""
    path = pathlib.Path(path)
    return path if path.is_absolute() or path.exists() else VERIFICATION_ROOT / path


def run_all(directory, args):
    """Run every deck under `directory`, each with its own table, then one line per deck.

    Returns [(name, study)], the study being None where that deck raised.
    """
    results = []
    for path in sorted(directory.glob("*/*.json")):
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
    """Load one deck, run its verification study, and write the plots it produced to disk."""
    deck = Deck.load(deck_path)
    study = ErrorConvergenceStudy(deck)
    study.run()
    write_plots(study, deck_path)
    return study


def write_plots(study, deck_path):
    """Save every figure in study.plots as <results root>/<deck's own path>_<plot name>.png."""
    try:
        relative = deck_path.resolve().relative_to(VERIFICATION_ROOT.resolve()).with_suffix("")
    except ValueError:
        relative = pathlib.Path(deck_path.stem)
    directory = RESULTS_ROOT / relative.parent
    directory.mkdir(parents=True, exist_ok=True)
    for name, figure in study.plots.items():
        figure.savefig(directory / f"{relative.name}_{name}.png")
        plt.close(figure)


def createScene(root):
    """The runSofa entry point, allowing to inspect a single mesh level of a deck in the GUI.

        runSofa -l SofaPython3 run.py --argv <deck> --argv <level>

    `<level>` indexes the deck's refinement sequence, so `0` is its coarsest mesh.
    """
    if len(sys.argv) != 3:
        sys.exit("run.py under runSofa expects: --argv <deck> --argv <level>")
    deck = Deck.load(resolve_path(sys.argv[1]))
    cells, _ = deck.levels()[int(sys.argv[2])]

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
            exclusive with ``deck``, and exactly one of the two is always set.
        traceback : bool
            Print the full stack for a deck that raises under ``--all``. False by default, where
            such a deck reports one line instead so the remaining decks still run.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    # Must choose either --all or pass the deck path
    which_decks = parser.add_mutually_exclusive_group(required=True)
    which_decks.add_argument("deck", nargs="?", help="path to a deck")
    which_decks.add_argument("--all", action="store_true", help="every deck under the verification root")
    parser.add_argument("--traceback", action="store_true",
                        help="full stack for a deck that raises under --all, which otherwise "
                             "reports one line so the remaining decks still run")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Run all the decks in verification
    if args.all:
        failed = [name for name, study in run_all(VERIFICATION_ROOT, args)
                  if study is None or not study.verified]
    # Run the prescribed deck case
    else:
        study = run(resolve_path(args.deck))
        failed = [] if study.verified else [study.name]

    # On failure, exit with error and report the problematic decks
    if failed:
        sys.exit(f"\nfailed: {', '.join(failed)}")
