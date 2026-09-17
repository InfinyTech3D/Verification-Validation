"""Runs an error-convergence-rate verification study in SOFA to exercise FEM Force Fields:

1. Solve a manufactured problem on a sequence of increasingly refined meshes.
2. Measure the discretisation error at each one.
3. Judge whether the observed order of accuracy matches what the discretisation should achieve.

References
----------
Roache, P. J. (1998). Verification and Validation in Computational Science and Engineering.
    Hermosa Publishers.
Roy, C. J. (2005). Review of Code and Solution Verification Procedures for Computational
    Simulation. Journal of Computational Physics, 205(1), 131-156.
Stern, F., Wilson, R. V., Coleman, H. W. and Paterson, E. G. (2001). Comprehensive Approach to
    Verification and Validation of CFD Simulations -- Part 1: Methodology and Procedures. Journal
    of Fluids Engineering, 123(4), 793-802.
"""

import sys
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

import Sofa.Core
import Sofa.Simulation

from .metrics import Measurement, METRICS
from .quadrature import MeshQuadrature
from .scene import MMSScene
from common.sofa.conventions import ELEMENTS


def _newton_diagnostics(newton):
    """Iterations, residual reduction and stopping status of the Newton solve at one level."""
    if newton is None:
        return None

    residuals = []
    for token in newton.residualGraph.value.split():
        try:
            residuals.append(float(token))
        except ValueError:      # the map key, not one of its residuals
            continue

    status = newton.status.value
    # ConvergedEquilibrium means the iterations never started because the residual was already zero.
    # On a loaded MMS scene that is not a success, it means the load never reached the system.
    converged = status.startswith("Converged") and status != "ConvergedEquilibrium"

    # Dimensionless reduction r/r0 of the residual. Undefined when it started at zero.
    reduction = float(np.sqrt(residuals[-1] / residuals[0])) if residuals and residuals[0] else float("nan")

    # Ignore the residual recorded once before iterating; floor at 0 if a crashed solver left none.
    return {"iterations": max(len(residuals) - 1, 0),
            "reduction": reduction,
            "status": "Converged" if converged else status}


def _paint(cell, accepted):
    """Colour a cell green when the settling test kept its rate, red when it discarded it."""
    GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"

    if not cell.strip() or not sys.stdout.isatty():
        return cell
    return f"{GREEN if accepted else RED}{cell}{RESET}"


def _mesh_label(cells):
    """This mesh's cell count per axis as text, e.g. [24, 24] -> "24x24"."""
    return 'x'.join(str(count) for count in cells)


def _print_progress_bar(level, level_count, label):
    """Print a bar showing how much of the run is done and which mesh it is on."""
    BAR_WIDTH, LABEL_WIDTH = 20, 14

    if not sys.stdout.isatty():
        return

    filled = round(BAR_WIDTH * level / level_count)
    bar = "=" * filled + "-" * (BAR_WIDTH - filled)
    progress = f"  {level}/{level_count} [{bar}] {label:<{LABEL_WIDTH}}"

    sys.stdout.write(f"\r{progress}\033[K")
    sys.stdout.flush()


def _clear_progress_bar():
    """Erase the progress bar so the table below starts on a clean line."""
    if sys.stdout.isatty():
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()


@dataclass
class ConvergenceRate:
    """The order observed between one mesh refinement level and the one below it, or why there is none."""

    class FailureReason(StrEnum):
        """Why computing a rate for a level pair failed, printed in its place when there is no rate."""
        FLOOR = "floor"   # one of the two errors has sunk into its own noise; the ratio is round-off, not order
        OSC = "osc"       # the error overshot then came back down (Stern, -1 < R < 0)
        DIV = "div"       # the error is diverging, steadily or through growing alternating increments (Stern, |R| >= 1)

    value: float | None
    reason: FailureReason | None = None


@dataclass
class MeshRefinementLevelResult:
    """One mesh of the refinement sequence: what was measured on it, and how its solve went."""

    label: str                                   # this mesh's name in the tables, e.g. "24x24"
    spacing: float                                # this mesh's step size, the h in the order formula
    elements_count: int                          # element count the mesh mapping actually produced
    values: dict                                 # metric name -> its measured error on this mesh
    noise_floor: dict                            # metric name -> the magnitude below which its error is noise
    newton: dict | None                          # this mesh's Newton solve diagnostics, or None if none ran
    rates: dict = field(default_factory=dict)    # metric name -> ConvergenceRate, unknown until this level is added


@dataclass
class MetricResult:
    """What the mesh refinement sequence found for one metric, once every level had been solved."""

    order: float | None     # the order at the finest settled mesh, or None if the order never settled
    accepted_levels: set    # level indices whose rate falls inside the settled range
    passed: bool            # whether `order` matches the deck's expected order, within tolerance


class ErrorConvergenceStudy:
    """Performs an error-convergence-rate study for a deck: builds a SOFA scene and solves it for
    progressively refined meshes.
    """

    def __init__(self, deck):
        self.name = deck.name
        self.deck = deck
        self.asymptotic_tolerance = deck.asymptotic_tolerance
        self.expected_order = deck.expected_order
        self.expected_order_tolerance = deck.expected_order_tolerance
        self.levels = []
        self.results = {}
        self.unconverged_levels = []
        self.verified = False

    def run(self):
        """Solve every mesh in the deck's refinement sequence, then evaluate and report the results."""
        mesh_refinement_levels = self.deck.levels()
        try:
            for index, (cells, spacing) in enumerate(mesh_refinement_levels, start=1):
                _print_progress_bar(index, len(mesh_refinement_levels), _mesh_label(cells))
                self.solve(cells, spacing)
        finally:
            _clear_progress_bar()
        self.evaluate_metrics()
        self.report()

    def solve(self, cells, spacing):
        """Build and solve one mesh's SOFA scene, then measure and add it as the next level.

        `cells` is that mesh's cell count per axis and `spacing` is its step size.
        """
        root = Sofa.Core.Node("root")
        mechanical = MMSScene(self.deck, cells).build(root).mechanical_node()
        Sofa.Simulation.init(root)
        Sofa.Simulation.animate(root, root.dt.value)

        self.measure(mechanical, _mesh_label(cells), spacing)

        Sofa.Simulation.unload(root)

    def measure(self, mechanical, label, spacing):
        """Measure an already-solved mesh, then add it as the next level.

        `mechanical` is that mesh's mechanical node, `label` names it and `spacing` is its step size.
        """
        element_kind = ELEMENTS[self.deck.element]
        nodes = mechanical.dofs.rest_position.array()
        u_h = mechanical.dofs.position.array() - nodes
        # node_indices[e] = the mesh-node indices forming element e (from the topology).
        node_indices = getattr(mechanical.topology, element_kind.data_name).array()

        quadrature = MeshQuadrature(nodes, node_indices, element_kind.cpp, self.deck.quadrature_degree)
        measurement = Measurement(quadrature, u_h, self.deck.manufactured_problem)

        level = MeshRefinementLevelResult(
            label=label, spacing=spacing,
            # The element count the mapping actually produced, since the deck states cells: one row
            # of node_indices per element, so 6x the cells for tetrahedra and 2x for triangles.
            elements_count=len(node_indices),
            values={metric.name: metric.measure(measurement) for metric in METRICS},
            noise_floor={metric.name: self.deck.relative_noise_floor * metric.scale(measurement)
                         for metric in METRICS},
            newton=_newton_diagnostics(getattr(mechanical, 'newton', None)))
        self.levels.append(level)
        level.rates = {metric.name: self.rate(metric.name) for metric in METRICS}

    def rate(self, metric_name):
        """The order across the two finest meshes so far, or None and why there is none."""
        if len(self.levels) < 2:
            return ConvergenceRate(None)

        current, previous = self.levels[-1], self.levels[-2]
        error, before = current.values[metric_name], previous.values[metric_name]
        # The discretization error should not be measured when below the round-off and/or iterative
        # solver error (Roy, 2005).
        if error <= current.noise_floor[metric_name] or before <= previous.noise_floor[metric_name]:
            return ConvergenceRate(None, ConvergenceRate.FailureReason.FLOOR)

        if len(self.levels) >= 3:
            # Stern's convergence ratio is read from the increments between three consecutive errors.
            # This also catches a sequence that oscillates while it is still decreasing on this pair.
            #
            #   R = (e_i - e_i-1) / (e_i-1 - e_i-2)
            #   e_i, e_i-1, e_i-2: the error on this level and the two coarser ones before it
            #   |R| >= 1:   divergence: steadily if R > 0, alternating in sign if R < 0
            #   -1 < R < 0: still converging, but oscillating
            #   0 <= R < 1: the normal case, falls through to the order formula below
            increment = error - before
            previous_increment = before - self.levels[-3].values[metric_name]
            ratio = increment / previous_increment if previous_increment else np.inf
            if abs(ratio) >= 1.0:
                return ConvergenceRate(None, ConvergenceRate.FailureReason.DIV)
            if ratio <= 0.0:
                return ConvergenceRate(None, ConvergenceRate.FailureReason.OSC)
        elif error >= before:
            # With 2 levels only the most that can be said is whether the error increased
            return ConvergenceRate(None, ConvergenceRate.FailureReason.DIV)

        return ConvergenceRate(np.log(error / before) / np.log(current.spacing / previous.spacing))

    def evaluate_metrics(self):
        """Judge the settled order per metric, then conclude self.verified: did this deck verify?"""
        self.results = {}
        for metric in METRICS:
            orders = [(index, level.rates[metric.name].value) for index, level in enumerate(self.levels)
                      if level.rates[metric.name].value is not None]

            # Contiguity matters: a readable order with a discarded pair beneath it does not extend a run.
            retained = []
            for index, order in reversed(orders):
                if retained and (index + 1 != retained[0][0]
                                 or abs(order - retained[0][1]) > self.asymptotic_tolerance):
                    break
                retained.insert(0, (index, order))
            # One order cannot demonstrate that anything settled -- it takes two to compare.
            if len(retained) < 2:
                retained = []

            if retained:
                order = retained[-1][1]
                accepted_levels = {index for index, _ in retained}
                # bool(): the comparison is on a numpy float, so it yields a numpy bool -- which is
                # not JSON-serializable, and would print as 1.0 wherever this gets recorded.
                passed = bool(abs(order - self.expected_order[metric.name]) <= self.expected_order_tolerance)
            else:
                order, accepted_levels, passed = None, set(), False

            self.results[metric.name] = MetricResult(order=order, accepted_levels=accepted_levels,
                                                      passed=passed)

        self.unconverged_levels = [level.label for level in self.levels
                                   if (level.newton["status"] if level.newton else None)
                                   not in ("Converged", None)]

        # A settled, passing order is not evidence of anything if the solve never converged.
        self.verified = not self.unconverged_levels and all(result.passed for result in self.results.values())

    def report(self):
        """Print one row per mesh refinement level, then the order each metric settled on."""
        LABEL_WIDTH, SPACING_WIDTH, ELEMENTS_WIDTH = 16, 20, 17
        RATE_WIDTH, ITERATIONS_WIDTH, RESIDUAL_WIDTH = 13, 14, 12

        header = (f"{'# cells/axis':>{LABEL_WIDTH}} {'mesh step size h':>{SPACING_WIDTH}}"
                  f" {'# of elements':>{ELEMENTS_WIDTH}}")
        for metric in METRICS:
            header += f" {metric.name:>{RATE_WIDTH}}"
        print(header + f" {'# Newton iters':>{ITERATIONS_WIDTH}} {'r/r0':>{RESIDUAL_WIDTH}}"
              f" Newton-Raphson status")

        for index, level in enumerate(self.levels):
            row = (f"{level.label:>{LABEL_WIDTH}} {level.spacing:>{SPACING_WIDTH}.5f}"
                   f" {level.elements_count:>{ELEMENTS_WIDTH}}")
            for metric in METRICS:
                rate = level.rates[metric.name]
                cell = f"{rate.value:.2f}" if rate.value is not None else (rate.reason or "")
                accepted = index in self.results[metric.name].accepted_levels
                row += f" {_paint(f'{cell:>{RATE_WIDTH}}', accepted)}"

            newton = level.newton
            if newton is None:
                row += f" {'':>{ITERATIONS_WIDTH}} {'':>{RESIDUAL_WIDTH}} no newton"
            else:
                row += (f" {newton['iterations']:>{ITERATIONS_WIDTH}}"
                        f" {newton['reduction']:>{RESIDUAL_WIDTH}.1e}"
                        f" {_paint(newton['status'], newton['status'] == 'Converged')}")
            print(row)

        # Summary rows, as (label, one text per metric, pass per metric or None where uncoloured).
        metric_results = [self.results[metric.name] for metric in METRICS]
        rows = [
            ("order p", [f"{r.order:.2f}" if r.order is not None else "not reached" for r in metric_results],
             [r.passed for r in metric_results]),
            ("expected", [f"{self.expected_order[metric.name]:.1f}" for metric in METRICS], None),
        ]

        print()
        for label, texts, passed in rows:
            row = f"{label:>{LABEL_WIDTH}} {'':>{SPACING_WIDTH}} {'':>{ELEMENTS_WIDTH}}"
            for index, text in enumerate(texts):
                padded = f"{text:>{RATE_WIDTH}}"
                row += f" {padded if passed is None else _paint(padded, passed[index])}"
            print(row)

        if self.unconverged_levels:
            print()
            print(f"  Newton did not converge at: {', '.join(self.unconverged_levels)}")


def overview(results):
    """Print one line per deck: the order each metric settled on, and whether its solves converged."""
    print()
    header = f"{'deck':<44}"
    for metric in METRICS:
        header += f" {metric.name:>9}"
    print(header + f" {'solver':>9}")

    for name, study in results:
        row = f"{name:<44}"
        for metric in METRICS:
            metric_result = None if study is None else study.results[metric.name]
            has_order = metric_result is not None and metric_result.order is not None
            cell = "error" if study is None else f"{metric_result.order:.2f}" if has_order else "--"
            row += f" {_paint(f'{cell:>9}', study is not None and metric_result.passed)}"
        solver = "error" if study is None else (
            "ok" if not study.unconverged_levels else f"{len(study.unconverged_levels)} bad")
        print(row + f" {_paint(f'{solver:>9}', solver == 'ok')}")
