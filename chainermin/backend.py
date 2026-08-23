import numpy

try:
    import cupy
except ImportError:
    cupy = None


def get_array_module(*args):
    """Return the array module (cupy or numpy) matching the given arrays."""
    return cupy.get_array_module(*args) if cupy is not None else numpy


def to_cpu(x):
    """Move an array to CPU (NumPy). No-op if already on CPU."""
    if cupy is not None and isinstance(x, cupy.ndarray):
        return x.get()
    return x


def to_gpu(x):
    """Move an array to GPU (CuPy). No-op if already on GPU."""
    if cupy is None:
        raise RuntimeError("CuPy is not installed; cannot move to GPU")
    if isinstance(x, cupy.ndarray):
        return x
    return cupy.asarray(x)
