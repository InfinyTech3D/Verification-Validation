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
        newtonDict = config.get('newton')
        if newtonDict is not None:
            node.addObject(newtonDict['type'], name='newton', **params(newtonDict))

        # Linear Solver & Preconditioner (optional)
        linearDict = config['linearSolver']
        precondDict = linearDict.get('preconditioner')
        if precondDict is None:
            node.addObject(linearDict['type'], name='linearSolver', **params(linearDict))
        else:
            precond_system_dict = precondDict['system']
            node.addObject(precond_system_dict['type'], name='precondSystem', **params(precond_system_dict))
            node.addObject(precondDict['type'], name='precond', linearSystem='@precondSystem',
                           **params(precondDict, 'system'))

            linear_system_dict = linearDict['system']
            node.addObject(linear_system_dict['type'], name='solverSystem',
                           preconditionerSystem='@precondSystem', **params(linear_system_dict))
            node.addObject(linearDict['type'], name='linearSolver', linearSystem='@solverSystem',
                           preconditioner='@precond', **params(linearDict, 'system', 'preconditioner'))

        # ODE Integration Scheme
        integration_sceme = params(config['integration'])
        if newtonDict is not None:
            integration_sceme['newtonSolver'] = '@newton'
        node.addObject(config['integration']['type'], name='ode', **integration_sceme)
