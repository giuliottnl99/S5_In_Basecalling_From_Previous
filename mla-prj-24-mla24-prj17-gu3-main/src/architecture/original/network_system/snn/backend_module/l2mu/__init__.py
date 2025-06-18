from src.architecture.utils import _lazy_import


def L2MULeaky(*args, **kwargs):
    return _lazy_import("src.architecture.original.network_system.snn.backend_module.l2mu.leaky", ".L2MU", "L2MU")(*args, **kwargs)


def L2MUSynaptic(*args, **kwargs):
    return _lazy_import("src.architecture.original.network_system.snn.backend_module.l2mu.synaptic", ".L2MU", "L2MU")(*args, **kwargs)
