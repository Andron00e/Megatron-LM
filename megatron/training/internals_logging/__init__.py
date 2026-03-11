# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Model internals logging module for Megatron-LM.

This module provides comprehensive logging of model internals during training,
including activation statistics, gradient statistics, relative weight updates
(delta_W), and angular metrics.

Example usage:
    from megatron.training.internals_logging import (
        InternalsLoggingConfig,
        InternalsHookManager,
        InternalsStateManager,
        InternalsLogger,
    )

    config = InternalsLoggingConfig.from_args(args)
    hook_manager = InternalsHookManager(config)
    state_manager = InternalsStateManager(weights_on_gpu=config.weights_on_gpu)
    logger = InternalsLogger(config, hook_manager, state_manager)

    # Register hooks on model
    hook_manager.register_hooks(model)

    # During training loop, at logging iterations:
    hook_manager.enable_capture()
    # ... forward pass ...
    logger.log_internals(model, iteration, wandb_writer)
    hook_manager.disable_capture()
"""

from .config import InternalsLoggingConfig
from .hooks import InternalsHookManager
from .state_manager import InternalsStateManager
from .logger import InternalsLogger

__all__ = [
    'InternalsLoggingConfig',
    'InternalsHookManager',
    'InternalsStateManager',
    'InternalsLogger',
]
