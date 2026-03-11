# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Hook management for capturing model internals during forward pass."""

from typing import Any, Callable, Dict, List, Tuple
import torch
from torch import Tensor
import torch.nn as nn

from .config import InternalsLoggingConfig

from megatron.core.transformer.identity_op import LoggingProbe


# Target linear layers within each TransformerLayer for delta_Y capture.
# Each entry is (parent_attr, linear_attr) relative to a TransformerLayer.
_LINEAR_LAYER_PATHS = [
    ("mlp", "linear_fc1"),
    ("mlp", "linear_fc2"),
    ("self_attention", "linear_qkv"),
    ("self_attention", "linear_proj"),
]


class InternalsHookManager:
    """Manages forward hooks for capturing model internals.

    This class registers PyTorch forward hooks on LoggingProbe modules placed
    throughout the model. LoggingProbe is an identity op that marks a capture
    point — the hook manager simply finds all probes and hooks them, requiring
    no hardcoded knowledge of the model architecture.

    Additionally, when delta_Y logging is enabled, registers hooks on target
    linear layers to capture their (input, output) pairs for re-running after
    the weight update.

    Attributes:
        config: Configuration for internals logging.
        hooks: List of registered hook handles.
        captured_activations: Dictionary mapping (logging_name, layer_number) tuples
            to captured activation tensors.
        captured_linear_io: Dictionary mapping (linear_name, layer_number) tuples
            to (input_tensor, output_tensor) pairs for delta_Y computation.
        should_capture: Flag indicating whether capture is currently enabled.
    """

    def __init__(self, config: InternalsLoggingConfig):
        """Initialize the hook manager.

        Args:
            config: Configuration specifying which layers/metrics to log.
        """
        self.config = config
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.captured_activations: Dict[Tuple[str, int], Tensor] = {}
        self.captured_linear_io: Dict[Tuple[str, int], Tuple[Tensor, Tensor]] = {}
        self.should_capture = False

    def register_hooks(self, model: nn.Module) -> None:
        """Register forward hooks in the model.

        Walks the module tree and registers a capture hook on every
        LoggingProbe whose layer_number passes the config filter.

        If delta_Y logging is enabled, also registers hooks on target linear layers 
        (linear_fc1, linear_fc2, linear_qkv, linear_proj) for delta_Y capture
        by capturng their (input, output) pairs during the forward pass.

        Args:
            model: The model to register hooks on.
        """
        from megatron.core.transformer.transformer_layer import TransformerLayer
        for _, module in model.named_modules():
            if isinstance(module, LoggingProbe):
                if self.config.should_log_layer(module.layer_number):
                    key = (module.logging_name, module.layer_number)
                    hook = module.register_forward_hook(
                        self._make_activation_hook(key)
                    )
                    self.hooks.append(hook)
            elif self.config.log_delta_y and isinstance(module, TransformerLayer):
                layer_number = module.layer_number
                if not self.config.should_log_layer(layer_number):
                    continue

                for parent_attr, linear_attr in _LINEAR_LAYER_PATHS:
                    parent = getattr(module, parent_attr, None)
                    if parent is None:
                        continue
                    linear_module = getattr(parent, linear_attr, None)
                    if linear_module is None:
                        continue

                    key = (linear_attr, layer_number)
                    hook = linear_module.register_forward_hook(
                        self._make_linear_io_hook(key)
                    )
                    self.hooks.append(hook)

    def _make_activation_hook(self, key: Tuple[str, int]) -> Callable:
        """Create a forward hook for capturing activations.

        Args:
            key: Tuple of (logging_name, layer_number) identifying this hook.

        Returns:
            Hook function that captures the module output.
        """
        def hook(
            module: nn.Module,
            input: Tuple[Tensor, ...],
            output: Any,
        ) -> None:
            if not self.should_capture:
                return

            # LoggingProbe is an identity op, so output is a plain tensor.
            if isinstance(output, Tensor):
                if key not in self.captured_activations:
                    self.captured_activations[key] = output

        return hook

    def _make_linear_io_hook(self, key: Tuple[str, int]) -> Callable:
        """Create a forward hook that captures (input, output) for a linear layer.

        Args:
            key: Tuple of (linear_name, layer_number) identifying this hook.

        Returns:
            Hook function that captures the input and output tensors.
        """
        def hook(
            module: nn.Module,
            input: Tuple[Tensor, ...],
            output: Any,
        ) -> None:
            if not self.should_capture:
                return
            if key in self.captured_linear_io:
                return  # Only capture the first micro-batch

            inp = input[0].detach().clone()

            if isinstance(output, tuple):
                out = output[0].detach().clone()
            else:
                out = output.detach().clone()

            self.captured_linear_io[key] = (inp, out)

        return hook

    def enable_capture(self) -> None:
        """Enable capture for the current iteration.

        Clears previous captured data and sets the capture flag.
        """
        self.should_capture = True
        self.captured_activations.clear()
        self.captured_linear_io.clear()

    def disable_capture(self) -> None:
        """Disable capture after logging."""
        self.should_capture = False

    def clear_captured_data(self) -> None:
        """Clear captured data to free memory."""
        self.captured_activations.clear()
        self.captured_linear_io.clear()

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __del__(self):
        """Cleanup hooks on deletion."""
        self.remove_hooks()
