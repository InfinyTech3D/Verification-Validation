"""Prefabs a study configures: they take the deck blocks they need and build themselves."""

import Sofa
import Sofa.Core


def params(config, *exclude):
    """Converts a set of parameters in a configuration dictionary to SOFA Data/parameterization.
    Every key is passed on except 'type' and excluded ones marked by the user through *exclude."""
    skip = ('type',) + exclude
    return {k: v for k, v in config.items() if k not in skip}


class ScenePrefab(Sofa.Prefab):
    """Base SOFA prefab: holds `configs` and offers adders a subclass calls to build itself from it."""

    def __init__(self, *args, configs=None, geometry=None, **kwargs):
        # object.__setattr__: the C++ object Sofa's own __setattr__ needs doesn't exist yet, and
        # Sofa.Prefab.__init__ below calls init(), which reads configs/geometry
        object.__setattr__(self, 'configs', configs or {})
        object.__setattr__(self, 'geometry', geometry)
        Sofa.Prefab.__init__(self, *args, **kwargs)

    def mechanical_node(self):
        raise NotImplementedError

    def add_force_field(self, node, config, material, template):
        """The *FEMForceField component under test, its Data merged from `config` and `material`."""
        node.addObject(config['type'], name='fem', template=template, topology='@topology',
                       **params(material), **params(config))

    def add_solvers(self, node, config):
        """Newton (if present) + linear solver (+ preconditioner) + integration scheme."""

        # Newton Solver
        newton_config = config.get('newton')
        if newton_config is not None:
            node.addObject(newton_config['type'], name='newton', **params(newton_config))

        # Linear Solver & Preconditioner (optional)
        linear_config = config['linearSolver']
        precond_config = linear_config.get('preconditioner')
        if precond_config is None:
            node.addObject(linear_config['type'], name='linearSolver', **params(linear_config))
        else:
            precond_system_config = precond_config['system']
            node.addObject(precond_system_config['type'], name='precondSystem', **params(precond_system_config))
            node.addObject(precond_config['type'], name='precond', linearSystem='@precondSystem',
                           **params(precond_config, 'system'))

            linear_system_config = linear_config['system']
            node.addObject(linear_system_config['type'], name='solverSystem',
                           preconditionerSystem='@precondSystem', **params(linear_system_config))
            node.addObject(linear_config['type'], name='linearSolver', linearSystem='@solverSystem',
                           preconditioner='@precond', **params(linear_config, 'system', 'preconditioner'))

        # ODE Integration Scheme
        integration_scheme = params(config['integration'])
        integration_scheme['linearSolver'] = '@linearSolver'
        if newton_config is not None:
            integration_scheme['newtonSolver'] = '@newton'
        node.addObject(config['integration']['type'], name='ode', **integration_scheme)
