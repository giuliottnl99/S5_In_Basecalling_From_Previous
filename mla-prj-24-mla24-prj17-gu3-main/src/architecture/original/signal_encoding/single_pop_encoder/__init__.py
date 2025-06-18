from src.architecture.utils import _lazy_import


def SinglePopEncoderLeaky(*args, **kwargs):
    return _lazy_import("src.architecture.original.signal_encoding.single_pop_encoder.leaky", ".input_encoder", "SinglePopEncoder")(
        *args, **kwargs)


def SinglePopEncoderSynaptic(*args, **kwargs):
    return _lazy_import("src.architecture.original.signal_encoding.single_pop_encoder.synaptic", ".input_encoder", "SinglePopEncoder")(
        *args, **kwargs)
