"""Generic SOFA scene abstraction. Scenes in VnV build off of it."""

from abc import ABC, abstractmethod


class SofaScene(ABC):
    """Abstract SOFA scene: loads its plugins, then builds the scene graph.

    A subclass appends the plugins it needs to `plugins`, and defines `body` to build the
    scene graph.
    """

    plugins = []  # plugins every scene needs; a subclass appends its own on top

    def __init__(self):
        # The list of plugins is kept as a copy per-instance.
        self.plugins = list(self.plugins)

    def add_plugins(self, root):
        """Add every plugin in `plugins` as one RequiredPlugin. Subclass appends to `self.plugins`."""
        root.addObject('RequiredPlugin', pluginName=self.plugins)

    @abstractmethod
    def body(self, root):
        """Build the scene graph."""

    def build(self, root):
        self.add_plugins(root)
        return self.body(root)
