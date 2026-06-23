"""Convenience functions to ensure reproducibility of results."""

import logging

import torch
from transformers import set_seed


def make_reproducible(seed: int, deterministic: bool = True) -> None:
    """Set seeds for various libraries."""
    logger = logging.getLogger("div")
    logger.info("Setting seed %s", seed)

    set_seed(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


#  LocalWords:  clm
