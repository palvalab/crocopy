# Re-export from the package-level _base module.
# This shim exists for backward compatibility — new code should import from crocopy._base.
from .._base import (  # noqa: F401
    T, NDArray, HAS_CUPY, get_module,
    supports_gpu, supports_multiprocessing,
)
