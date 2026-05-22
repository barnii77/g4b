from cuda.bindings import runtime as cudart


def runtime_error(msg):
    """A wrapper for `raise RuntimeError` that can be used in an expression."""
    raise RuntimeError(msg)


class CudaError(RuntimeError):
    pass


def cuda_check(err: cudart.cudaError_t | tuple):
    """Anyone who's ever used CUDA knows what this is ;)"""
    if isinstance(err, tuple):
        err = err[0]
    assert isinstance(err, cudart.cudaError_t)
    if err != cudart.cudaError_t.cudaSuccess:
        raise CudaError(cudart.cudaGetErrorString(err))
