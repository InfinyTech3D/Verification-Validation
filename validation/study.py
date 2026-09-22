"""ISFComparisonStudy: wraps one validation case's sofa_scene/fenics_scene pair."""

import importlib.util
import json
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from validation.metrics import METRICS

VALIDATION_ROOT = pathlib.Path(__file__).parent
RESULTS_ROOT = VALIDATION_ROOT / "results"


def _load_module(path, name):
    """Import the .py file at `path`"""
    sys.modules.pop('case', None)  # each sofa_scene.py does `from case import ...`; evict the stale one

    spec = importlib.util.spec_from_file_location(name, path)  # `name` unique per case: avoids the same collision for this module itself
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ISFComparisonStudy:
    """Runs one validation case's SOFA scene against recorded references from other software.

    If an analytical solution is provided, it is used.
    """

    def __init__(self, case_dir):
        self.case_dir = pathlib.Path(case_dir)
        self.name = self.case_dir.name
        self.results = {}
        self.analytic_results = {}
        self.analytic_passed = {}
        self.max = {}
        self.passed = {}
        self.plots = {}

    def run(self, regenerate_fenics=False):
        with open(self.case_dir / 'deck.json') as f:
            deck = json.load(f)
        element_decks = [deck] if 'geometry' in deck else list(deck.values())

        reference_file = self.case_dir / 'reference_solution.json'

        for deck in element_decks:
            element = deck['element']

            reference = json.loads(reference_file.read_text()) if reference_file.exists() else {}
            has_fenics = element in reference and 'fenics' in reference[element]
            if regenerate_fenics or not has_fenics:
                fenics_scene = _load_module(self.case_dir / 'fenics_scene.py', f'fenics_scene_{self.name}')
                fenics_scene.main(deck)
                reference = json.loads(reference_file.read_text())

            reference = reference[element]['fenics']
            x_fenics, u_fenics = np.array(reference['x']), np.array(reference['u'])

            sofa_scene = _load_module(self.case_dir / 'sofa_scene.py', f'sofa_scene_{self.name}')
            x_sofa, u_sofa = sofa_scene.solve(deck)

            assert np.allclose(x_sofa, x_fenics, atol=1e-8), \
                "SOFA and FEniCS meshes disagree on node coordinates: comparison is not meaningful"

            self.results[element] = {metric.name: metric.measure(u_sofa, u_fenics) for metric in METRICS}
            passed = all(self.results[element][name] < deck['tolerance'][name]
                        for name in self.results[element])

            u_exact = None
            analytic_path = self.case_dir / 'analytic_solution.py'
            if analytic_path.exists():
                analytic_solution = _load_module(analytic_path, f'analytic_solution_{self.name}')
                u_exact = analytic_solution.displacement(x_sofa, deck)
                self.analytic_results[element] = {metric.name: metric.measure(u_sofa, u_exact)
                                                  for metric in METRICS}
                self.analytic_passed[element] = all(
                    self.analytic_results[element][name] < deck['toleranceAnalytic'][name]
                    for name in self.analytic_results[element])
                passed = passed and self.analytic_passed[element]

            self.passed[element] = passed

            # Max RMS value
            max_scale = np.max(np.abs(u_fenics)) or 1.0
            self.max[element] = np.max(np.abs(u_sofa - u_fenics)) / max_scale

            self.plots[element] = self._plot(element, x_sofa, u_sofa, x_fenics, u_fenics, u_exact)
        return self

    def write_results(self):
        """Save the figures this study produced under <results root>/<case's own path>/."""
        try:
            relative = self.case_dir.resolve().relative_to(VALIDATION_ROOT.resolve())
        except ValueError:
            relative = pathlib.Path(self.case_dir.name)
        directory = RESULTS_ROOT / relative
        directory.mkdir(parents=True, exist_ok=True)

        for name, figure in self.plots.items():
            figure.savefig(directory / f"{name}.png")
            plt.close(figure)

    def _plot(self, element, x_sofa, u_sofa, x_fenics, u_fenics, u_exact):
        figure, axes = plt.subplots(figsize=(7, 5))
        if u_exact is not None:
            axes.plot(x_sofa, u_exact, 'k--', lw=1, label='analytic')
        axes.plot(x_fenics, u_fenics, 's', mfc='none', mec='tab:blue', ms=8, mew=1.5, label='FEniCS')
        axes.plot(x_sofa, u_sofa, '+', color='tab:red', ms=8, mew=1.5, label='SOFA')
        axes.set_xlabel('x')
        axes.set_ylabel('u')
        axes.set_title(f'{self.name}/{element}')
        axes.legend()
        axes.grid(alpha=.3)
        axes.annotate(f"rms = {self.results[element]['rms']:.2e}", xy=(.04, .92),
                      xycoords='axes fraction', fontsize=9, color='dimgray')
        return figure


def _paint(cell, accepted):
    """Colour a cell green when accepted, red otherwise."""
    GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"

    if not cell.strip() or not sys.stdout.isatty():
        return cell
    return f"{GREEN if accepted else RED}{cell}{RESET}"


def overview(results):
    """Print one line per case per element: its RMS and max relative error against the reference
    solution, and its RMS against the analytic solution when the case has one."""
    name_width = max(len('case'), *(len(name) for name, _ in results))
    element_width = max(len('element'), *(len(element) for _, study in results if study is not None
                                          for element in study.results))

    print()
    print(f"{'case':<{name_width}} {'element':<{element_width}} "
          f"{'rms':>12} {'max':>12} {'analytic rms':>14}")

    for name, study in results:
        if study is None:
            error = "error"
            print(f"{name:<{name_width}} {'--':<{element_width}} "
                  f"{_paint(f'{error:>12}', False)} {_paint(f'{error:>12}', False)} "
                  f"{_paint(f'{error:>14}', False)}")
            continue

        for element, metrics in study.results.items():
            rms_text, max_text = f"{metrics['rms']:.3e}", f"{study.max[element]:.3e}"
            analytic = study.analytic_results.get(element)
            if analytic is None:
                analytic_text = f"{'--':>14}"
            else:
                analytic_text = _paint(f"{analytic['rms']:>14.3e}", study.analytic_passed[element])
            print(f"{name:<{name_width}} {element:<{element_width}} "
                  f"{_paint(f'{rms_text:>12}', study.passed[element])} "
                  f"{max_text:>12} {analytic_text}")
