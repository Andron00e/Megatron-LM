# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Configuration for model internals logging."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class InternalsLoggingConfig:
    """Configuration for model internals logging.

    Attributes:
        enabled: Master switch to enable/disable internals logging.
        log_activation_stats: Log activation statistics (mean, std, min, max, kurtosis).
        log_gradient_stats: Log per-layer gradient statistics.
        log_relative_updates: Log relative weight changes (delta_W).
        log_angular_metrics: Log angular updates and gradient-weight alignment.
        layers_to_log: List of layer indices to log, or None for all layers.
        log_interval: How often to log (in iterations).
    """
    enabled: bool = False
    log_activation_stats: bool = True
    log_gradient_stats: bool = True
    log_relative_updates: bool = True
    log_angular_metrics: bool = True
    log_delta_y: bool = False
    layers_to_log: Optional[List[int]] = None
    log_interval: int = 1
    weights_on_gpu: bool = False

    @classmethod
    def from_args(cls, args) -> 'InternalsLoggingConfig':
        """Create config from command-line arguments."""
        layers = None
        if hasattr(args, 'internals_log_layers') and args.internals_log_layers != 'all':
            try:
                layers = [int(x.strip()) for x in args.internals_log_layers.split(',')]
            except ValueError:
                layers = None

        return cls(
            enabled=getattr(args, 'log_model_internals', False),
            log_activation_stats=getattr(args, 'log_activation_stats', False),
            log_gradient_stats=getattr(args, 'log_gradient_stats', False),
            log_relative_updates=getattr(args, 'log_relative_updates', False),
            log_angular_metrics=getattr(args, 'log_angular_metrics', False),
            log_delta_y=getattr(args, 'log_delta_y', False),
            layers_to_log=layers,
            log_interval=getattr(args, 'log_interval', 1),
            weights_on_gpu=getattr(args, 'internals_weights_on_gpu', False),
        )

    def should_log_layer(self, layer_num: int) -> bool:
        """Check if a specific layer should be logged."""
        if self.layers_to_log is None:
            return True
        return layer_num in self.layers_to_log
