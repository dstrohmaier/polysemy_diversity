"""Convenience functions to ensure reproducibility of results."""

import logging
import os

import torch
from transformers import set_seed

# cuBLAS needs a fixed workspace to make its GEMMs deterministic, and it reads this
# variable when CUDA initialises -- by the time we could set it here, that has
# usually already happened. The justfile exports it; this is the value it uses.
_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def make_reproducible(seed: int, deterministic: bool = True) -> None:
    """Set seeds for various libraries."""
    logger = logging.getLogger("div")
    logger.info("Setting seed %s", seed)

    set_seed(seed)

    if deterministic:
        # Without the workspace config, use_deterministic_algorithms turns the first
        # CUDA matmul into an opaque cuBLAS RuntimeError. Fail here instead, where the
        # cause and the fix are obvious.
        if torch.cuda.is_available() and not os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
            raise RuntimeError(
                "Deterministic mode on CUDA requires CUBLAS_WORKSPACE_CONFIG to be "
                f"set before the process starts (e.g. "
                f"CUBLAS_WORKSPACE_CONFIG={_CUBLAS_WORKSPACE_CONFIG}). The justfile "
                "recipes export it; export it too when running the script directly."
            )
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


#  LocalWords:  clm
