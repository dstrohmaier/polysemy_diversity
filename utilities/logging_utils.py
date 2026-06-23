"""Utilities for logging that can be shared."""

import logging

from pathlib import Path

import torch


def start_logging(
    logging_directory: Path = Path("logs"),
    logger_name: str = "div",
    file_name: str = "run.log",
):
    """Set up logger to be shared across files."""
    logging_directory.mkdir(parents=True, exist_ok=True)

    logging_fpath = logging_directory / file_name
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(logging_fpath)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s:" "%(levelname)s:" "%(filename)s: " "%(message)s"
    )
    file_handler.setFormatter(formatter)

    for old_fh in logger.handlers[:]:  # remove all old handlers (copy: removeHandler mutates the list)
        logger.removeHandler(old_fh)
    logger.addHandler(file_handler)  # set the new handler
    logger.propagate = False  # file handler is authoritative; avoid duplicate emit via root

    logger.info("Started running")
    return logger


def log_model_device(model: torch.nn.Module, logger_name: str = "clm") -> None:
    """Log all devices for torch Module."""
    logger = logging.getLogger(logger_name)

    devices = set()

    for name, parameter in model.named_parameters():
        devices.add(parameter.device)

    if len(devices) == 1:
        (device,) = devices
        logger.debug("Model parameters on device %s", device)
        return

    for name, parameter in model.named_parameters():
        logger.debug("Model parameter %s on device %s", name, parameter.device)


def get_model_size(model: torch.nn.Module) -> float:
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    mb_size = (param_size + buffer_size) / (1024**2)
    return mb_size


def get_model_param_count(model: torch.nn.Module) -> int:
    param_count = 0
    for param in model.parameters():
        param_count += param.nelement()

    return param_count
