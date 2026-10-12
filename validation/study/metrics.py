"""The quantities a comparison study measures: one class per metric, so a metric is defined in one place."""

from abc import ABC, abstractmethod

import numpy as np


class Metric(ABC):
    """A quantity measured between a SOFA solve and a reference solved by another software."""

    name = ""

    @abstractmethod
    def measure(self, u_sofa, u_fenics):
        """The metric's value for one case."""


class RMSRelative(Metric):
    """Root-mean-square of the difference, relative to the RMS of the reference:

        rms = sqrt(mean((u_sofa - u_fenics)^2)) / sqrt(mean(u_fenics^2))
    """

    name = "rms"

    def measure(self, u_sofa, u_fenics):
        diff = u_sofa - u_fenics
        scale = np.sqrt(np.mean(u_fenics ** 2)) or 1.0
        return np.sqrt(np.mean(diff ** 2)) / scale


# METRICS is the fixed list of metric instances every case is measured against.
METRICS = [RMSRelative()]
