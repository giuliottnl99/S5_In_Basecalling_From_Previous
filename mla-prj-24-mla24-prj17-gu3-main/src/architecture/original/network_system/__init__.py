from src.architecture.utils import _lazy_import


def LMU(*args, **kwargs):
    return _lazy_import("src.architecture.original.network_system.ann", ".lmu", "LMUNet")(*args, **kwargs)


def L2MU(*args, **kwargs):
    return _lazy_import("src.architecture.original.network_system.snn", ".l2mu", "L2MUNet")(*args, **kwargs)
