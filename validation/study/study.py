"""ISFComparisonStudy: wraps one validation case's sofa_scene/fenics_scene pair."""

import importlib.util
import json
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .metrics import METRICS

CASES_ROOT = pathlib.Path(__file__).parent.parent / "cases"
RESULTS_ROOT = pathlib.Path(__file__).parent.parent / "results"


def _load_module(path, name):
    """Import the .py file at `path`"""
    spec = importlib.util.spec_from_file_location(name, path)  # `name` unique per case: avoids a module collision across cases
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
        self.solutions = {}

    def run(self, regenerate_fenics=False):
        with open(self.case_dir / 'case.json') as f:
            case = json.load(f)
        element_cases = [case] if 'geometry' in case else list(case.values())

        reference_file = self.case_dir / 'reference_solution.json'

        for case in element_cases:
            element = case['element']

            reference = json.loads(reference_file.read_text()) if reference_file.exists() else {}
            has_fenics = element in reference and 'fenics' in reference[element]
            if regenerate_fenics or not has_fenics:
                fenics_scene = _load_module(self.case_dir / 'fenics_scene.py', f'fenics_scene_{self.name}')
                fenics_scene.main(case)
                reference = json.loads(reference_file.read_text())

            reference = reference[element]['fenics']
            x_fenics, u_fenics = np.array(reference['x']), np.array(reference['u'])

            sofa_scene = _load_module(self.case_dir / 'sofa_scene.py', f'sofa_scene_{self.name}')
            x_sofa, u_sofa, polygons = sofa_scene.solve(case)
            self.solutions[element] = (x_sofa, u_sofa)

            assert np.allclose(x_sofa, x_fenics, atol=1e-8), \
                "SOFA and FEniCS meshes disagree on node coordinates: comparison is not meaningful"

            self.results[element] = {metric.name: metric.measure(u_sofa, u_fenics) for metric in METRICS}
            passed = all(self.results[element][name] < case['tolerance'][name]
                        for name in self.results[element])

            u_exact = None
            analytic_path = self.case_dir / 'analytic_solution.py'
            if analytic_path.exists():
                analytic_solution = _load_module(analytic_path, f'analytic_solution_{self.name}')
                u_exact = analytic_solution.displacement(x_sofa, case)
                self.analytic_results[element] = {metric.name: metric.measure(u_sofa, u_exact)
                                                  for metric in METRICS}
                self.analytic_passed[element] = all(
                    self.analytic_results[element][name] < case['toleranceAnalytic'][name]
                    for name in self.analytic_results[element])
                passed = passed and self.analytic_passed[element]

            self.passed[element] = passed

            # Max RMS value
            max_scale = np.max(np.abs(u_fenics)) or 1.0
            self.max[element] = np.max(np.abs(u_sofa - u_fenics)) / max_scale

            self.plots[element] = self._plot(element, x_sofa, u_sofa, polygons, u_fenics, u_exact)
        return self

    def write_results(self):
        """Save the figures this study produced under <results root>/<case's own path>/."""
        try:
            relative = self.case_dir.resolve().relative_to(CASES_ROOT.resolve())
        except ValueError:
            relative = pathlib.Path(self.case_dir.name)
        directory = RESULTS_ROOT / relative
        directory.mkdir(parents=True, exist_ok=True)

        for name, figure in self.plots.items():
            figure.savefig(directory / f"{name}.png")
            plt.close(figure)

    def _plot(self, element, x, u_sofa, polygons, u_fenics, u_exact):
        """`polygons` are the elements in 1D and 2D, and the boundary faces in 3D."""
        if x.ndim == 1:
            figure, axes = plt.subplots(figsize=(7, 5))
            if u_exact is not None:
                plot_1d(axes, x, u_exact, polygons, 'k--', lw=1, label='analytic')
            plot_1d(axes, x, u_fenics, polygons, 's', mfc='none', mec='tab:blue', ms=8, mew=1.5, label='FEniCS')
            plot_1d(axes, x, u_sofa, polygons, '+', color='tab:red', ms=8, mew=1.5, label='SOFA')
            axes.set_xlabel('x')
            axes.set_ylabel('u')
            axes.set_title(f'{self.name}/{element}')
            axes.legend()
            axes.grid(alpha=.3)
            axes.annotate(f"rms = {self.results[element]['rms']:.2e}", xy=(.04, .92),
                          xycoords='axes fraction', fontsize=9, color='dimgray')
            return figure

        sources = [('SOFA', u_sofa), ('FEniCS', u_fenics)]
        if u_exact is not None:
            sources.append(('Analytic', u_exact))
        dim = x.shape[1]
        components = ['u_x', 'u_y', 'u_z'][:dim]

        extent = np.ptp(x, axis=0)
        panel_width = 4.5
        panel_height = panel_width * extent[1] / extent[0] + 0.5 if dim == 2 else 3.5
        figure, axes = plt.subplots(len(sources), len(components), squeeze=False,
                                    figsize=(panel_width * len(components) + 0.5,
                                             panel_height * len(sources) + 1.2),
                                    layout='constrained', subplot_kw={'projection': '3d'} if dim == 3 else {})
        plot = plot_2d if dim == 2 else plot_3d

        for column, component in enumerate(components):
            # Shared colour scale per component, so the solvers compare at a glance.
            values = [u[:, column] for _, u in sources]
            vmin, vmax = min(v.min() for v in values), max(v.max() for v in values)
            # A component constant over the plot would stretch round-off over the colour map.
            span = 1e-6 * max(np.abs(u).max() for _, u in sources)
            if vmax - vmin < span:
                vmin, vmax = (vmin + vmax - span) / 2, (vmin + vmax + span) / 2
            for row, (label, u) in enumerate(sources):
                image = plot(axes[row, column], x, u, u[:, column], polygons, cmap='viridis',
                             norm=plt.Normalize(vmin, vmax))
                axes[row, column].set_title(f'{label}: ${component}$')
            colorbar = figure.colorbar(image, ax=axes[:, column], location='bottom', shrink=0.8)
            colorbar.ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
            colorbar.ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

        figure.suptitle(f"{self.name}/{element}    rms = {self.results[element]['rms']:.2e}")
        return figure


def plot_1d(ax, x, values, edges, *args, **kwargs):
    """Nodal `values` along the `edges`, as one line broken between edges."""
    gap = np.full((len(edges), 1), np.nan)
    return ax.plot(np.hstack([x[edges], gap]).ravel(), np.hstack([values[edges], gap]).ravel(), *args, **kwargs)


def plot_2d(ax, x, u, values, elements, **kwargs):
    """Colour map of nodal `values` over the triangles or quads `elements`, deformed by `u`; each takes the mean
    of its nodes."""
    deformed = x + u
    collection = ax.add_collection(PolyCollection(deformed[elements], array=values[elements].mean(axis=1), **kwargs))
    ax.autoscale_view()
    ax.set_aspect('equal')
    return collection


def plot_3d(ax, x, u, values, faces, **kwargs):
    """Colour map of nodal `values` over the boundary `faces`, deformed by `u`, y upwards; each takes the mean
    of its nodes."""
    view = (x + u)[:, [0, 2, 1]]
    collection = ax.add_collection3d(Poly3DCollection(view[faces], array=values[faces].mean(axis=1), **kwargs))
    ax.auto_scale_xyz(*view.T)
    ax.set_box_aspect(np.ptp(view, axis=0))
    ax.set(xlabel='x', ylabel='z', zlabel='y')
    return collection


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
