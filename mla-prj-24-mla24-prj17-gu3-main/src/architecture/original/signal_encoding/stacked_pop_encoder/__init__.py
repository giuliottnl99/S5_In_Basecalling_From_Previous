from src.architecture.utils import _lazy_import


def StackedPopEncoderLeaky(*args, **kwargs):
    return _lazy_import("src.architecture.original.signal_encoding.stacked_pop_encoder.leaky",
                        ".input_encoder", "StackedPopEncoder")(*args, **kwargs)


def StackedPopEncoderSynaptic(*args, **kwargs):
    return _lazy_import("src.architecture.original.signal_encoding.stacked_pop_encoder.synaptic",
                        ".input_encoder", "StackedPopEncoder")(*args, **kwargs)
