# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Metric computation functions for model internals logging."""

import math
from typing import Dict, Optional, Tuple
import torch
from torch import Tensor
import torch.nn as nn


def _extract_layer_index(param_name: str) -> Optional[int]:
    """Extract transformer layer index from a dotted parameter name.

    E.g. 'decoder.layers.3.self_attention.linear_qkv.weight' → 3.
    Returns None if no layer index is found.
    """
    parts = param_name.split('.')
    for i, part in enumerate(parts):
        if part == 'layers' and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return None
    return None


def compute_activation_stats(tensor: Tensor) -> Dict[str, float]:
    """Compute activation statistics per vector along the last dimension.

    For a tensor of shape [..., d], each slice along the last dimension is
    one activation vector. Returns the average of per-vector statistics
    (mean, std, kurtosis, min, max, rms).

    min/max are the expectation of per-vector extremes: min(-1).mean() and
    max(-1).mean(). In distributed training, all stats are AVG-reduced
    across DP groups.

    Args:
        tensor: Activation tensor of shape [..., d].

    Returns:
        Dictionary with mean, std, min, max, kurtosis.
    """
    with torch.no_grad():
        d = tensor.shape[-1]
        tensor_2d = tensor.reshape(-1, d).float()  # [N, d]

        mean = tensor_2d.mean()

        var = tensor_2d.pow(2).mean()

        act_rms = tensor_2d.pow(2).mean(-1).sqrt() # [N]
        act_rms = act_rms.mean()

        tensor_min = tensor_2d.min(-1).values.mean()
        tensor_max = tensor_2d.max(-1).values.mean()

        # kurtosis
        act_rms_per_neuron = tensor_2d.pow(2).mean(0).sqrt() # [d]
        denom = act_rms_per_neuron.pow(2).mean().pow(2)
        kurtosis = act_rms_per_neuron.pow(4).mean() / (denom + 1e-8)

        # all_rms = (tensor_2d**2).mean().sqrt()
        # normed_acts = tensor_2d / (all_rms + 1e-8)
        # alt_kurtosis = (normed_acts**2).mean(0).var()
        # print(f"Kurtosis: {kurtosis.item()} versus {alt_kurtosis.item()}")

    return {
        'mean': mean.item(),
        'var': var.item(),
        'min': tensor_min.item(),
        'max': tensor_max.item(),
        'kurtosis': kurtosis.item(),
        'rms': act_rms.item(),
    }


def get_param_grad(param: Tensor) -> Optional[Tensor]:
    """Get gradient tensor for a parameter.

    Megatron-LM uses main_grad for distributed training instead of standard grad.
    This helper checks both.

    Args:
        param: Parameter tensor.

    Returns:
        Gradient tensor or None if no gradient is available.
    """
    # Check main_grad first (used by Megatron's DistributedDataParallel)
    if hasattr(param, 'main_grad') and param.main_grad is not None:
        return param.main_grad
    # Fall back to standard grad
    if param.grad is not None:
        return param.grad
    return None


def compute_gradient_norm(param: Tensor) -> float:
    """Compute L2 norm of gradient for a parameter.

    Args:
        param: Parameter tensor with .grad or .main_grad attribute.

    Returns:
        L2 norm of the gradient, or 0.0 if no gradient.
    """
    grad = get_param_grad(param)
    if grad is None:
        return 0.0
    with torch.no_grad():
        return grad.float().norm().item()


def compute_weight_delta_per_neuron(
    current: Tensor,
    previous: Tensor,
) -> Dict[str, float]:
    """Compute per-neuron (per-row) relative weight changes.

    For a weight matrix W of shape [out_features, in_features], computes
    the relative change for each output neuron (row) and returns statistics.

    Args:
        current: Current weight tensor (at least 2D).
        previous: Previous weight tensor (same shape as current).

    Returns:
        Dictionary with per-neuron statistics:
        - 'full': Relative change of full matrix (Frobenius norm)
        - 'per_neuron_mean': Mean of per-neuron relative changes
        - 'per_neuron_std': Std of per-neuron relative changes
        - 'per_neuron_max': Max per-neuron relative change
        - 'per_neuron_min': Min per-neuron relative change
    """
    with torch.no_grad():
        result = {'full': _relative_norm_change(current, previous)}

        # Per-neuron stats only make sense for 2D+ tensors
        if current.dim() >= 2:
            if previous.device != current.device:
                previous = previous.to(current.device)

            num_neurons = current.size(0)
            current_2d = current.float().view(num_neurons, -1)
            previous_2d = previous.float().view(num_neurons, -1)

            delta_per_neuron = (current_2d - previous_2d).norm(dim=1)  # [num_neurons]
            prev_per_neuron = previous_2d.norm(dim=1)  # [num_neurons]

            mask = prev_per_neuron > 1e-10
            relative_delta = torch.zeros_like(delta_per_neuron)
            relative_delta[mask] = delta_per_neuron[mask] / prev_per_neuron[mask]

            result['per_neuron_mean'] = relative_delta.mean().item()
            # result['per_neuron_std'] = relative_delta.std().item() if num_neurons > 1 else 0.0
            # result['per_neuron_max'] = relative_delta.max().item()
            # result['per_neuron_min'] = relative_delta.min().item()

        return result


def _cosine_similarity(a: Tensor, b: Tensor):
    """Compute cosine similarity between two flat tensors.

    Returns (cos_sim_tensor, norm_a, norm_b) or None if either norm is near zero.
    """
    a_norm = a.norm()
    b_norm = b.norm()
    if a_norm < 1e-10 or b_norm < 1e-10:
        return None
    cos = (a @ b) / (a_norm * b_norm)
    return cos.clamp(-1.0, 1.0), a_norm, b_norm


def compute_angular_update(
    current: Tensor,
    previous: Tensor,
) -> Dict[str, float]:
    """Compute angular change between weight tensors.

    Measures how much the *direction* of weights changed, independent of magnitude.
    Useful for analyzing spherical/normalized training dynamics (nGPT, EDM2).

    Args:
        current: Current weight tensor.
        previous: Previous weight tensor (same shape as current).

    Returns:
        Dictionary with:
        - cos_similarity: cosine similarity between W_t and W_{t-1} (-1 to 1)
        - angular_change: angle in radians (0 to pi)
        - angular_change_degrees: angle in degrees (0 to 180)
    """
    with torch.no_grad():
        if previous.device != current.device:
            previous = previous.to(current.device)

        result = _cosine_similarity(current.flatten().float(), previous.flatten().float())
        if result is None:
            return {
                'cos_similarity': 0.0,
                'angular_change': 0.0,
                'angular_change_degrees': 0.0,
            }

        cos_sim, _, _ = result
        angular_change = torch.acos(cos_sim).item()

    return {
        'cos_similarity': cos_sim.item(),
        'angular_change': angular_change,
        'angular_change_degrees': angular_change * 180.0 / math.pi,
    }


def compute_gradient_weight_alignment(param: Tensor) -> Dict[str, float]:
    """Compute alignment between gradient and weight vectors.

    Measures whether gradients point along weights (radial) or orthogonal (tangential).
    For spherical training, tangential updates (cos ~ 0) preserve weight norms.

    Args:
        param: Parameter tensor with .grad or .main_grad attribute.

    Returns:
        Dictionary with:
        - cos_alignment: cosine of angle between gradient and weight (-1 to 1)
          - cos ~ 1: gradient increases weight norm (radial, norm-increasing)
          - cos ~ 0: gradient changes direction only (tangential)
          - cos ~ -1: gradient decreases weight norm (radial, norm-decreasing)
        - radial_component: ||grad|| * cos(grad, W) - component along weight direction
        - tangential_component: ||grad|| * sin(grad, W) - component orthogonal to weight
    """
    _ZERO = {'cos_alignment': 0.0, 'radial_component': 0.0, 'tangential_component': 0.0}

    grad = get_param_grad(param)
    if grad is None:
        return _ZERO

    with torch.no_grad():
        result = _cosine_similarity(grad.flatten().float(), param.data.flatten().float())
        if result is None:
            return _ZERO

        cos_align, grad_norm, _ = result
        radial = (grad_norm * cos_align).item()
        tangential = (grad_norm * torch.sqrt(1 - cos_align ** 2)).item()

    return {
        'cos_alignment': cos_align.item(),
        'radial_component': radial,
        'tangential_component': tangential,
    }


_QKV_NAMES = ('Q', 'K', 'V')


def _split_qkv_dim(
    tensor: Tensor,
    num_query_groups: int,
    num_query_heads_per_group: int,
    head_dim: int,
    qkv_dim: int,
) -> tuple:
    """Reshape and split a fused QKV tensor into Q, K, V components.

    The qkv_dim indexes the fused dimension of size ng*(nq+2)*hn.
    That dimension is replaced by [ng, component_dim] and then split.

    Use qkv_dim=-1 for activation tensors [*, total_qkv] and
    qkv_dim=0 for weight tensors [total_qkv, in_features].

    Returns:
        (Q, K, V) tuple of tensors.
    """
    qkv_dim = qkv_dim % tensor.dim()
    group_size = (num_query_heads_per_group + 2) * head_dim

    shape = list(tensor.shape)
    shape[qkv_dim:qkv_dim + 1] = [num_query_groups, group_size]
    reshaped = tensor.reshape(shape)

    split_sizes = [num_query_heads_per_group * head_dim, head_dim, head_dim]
    return torch.split(reshaped, split_sizes, dim=qkv_dim + 1)


def compute_weight_delta_qkv_split(
    current: Tensor,
    previous: Tensor,
    num_query_groups: int,
    num_query_heads_per_group: int,
    head_dim: int,
) -> Dict[str, Dict[str, float]]:
    """Split a fused QKV weight into Q, K, V and compute delta_W per component.

    Returns:
        Dict mapping component name ('Q', 'K', 'V') to delta stats dicts.
    """
    with torch.no_grad():
        if previous.device != current.device:
            previous = previous.to(current.device)

        in_features = current.shape[-1]
        qkv_args = (num_query_groups, num_query_heads_per_group, head_dim)
        curr_comps = _split_qkv_dim(current, *qkv_args, qkv_dim=0)
        prev_comps = _split_qkv_dim(previous, *qkv_args, qkv_dim=0)

        return {
            name: compute_weight_delta_per_neuron(
                c.reshape(-1, in_features), p.reshape(-1, in_features),
            )
            for name, c, p in zip(_QKV_NAMES, curr_comps, prev_comps)
        }


def compute_weight_delta_swiglu_split(
    current: Tensor,
    previous: Tensor,
) -> Dict[str, Dict[str, float]]:
    """Split a fused SwiGLU fc1 weight into gate and up components and compute delta_W.

    Returns:
        Dict mapping component name ('gate', 'up') to delta stats dicts.
    """
    with torch.no_grad():
        if previous.device != current.device:
            previous = previous.to(current.device)

        curr_gate, curr_up = torch.chunk(current, 2, dim=0)
        prev_gate, prev_up = torch.chunk(previous, 2, dim=0)

        return {
            'gate': compute_weight_delta_per_neuron(curr_gate, prev_gate),
            'up': compute_weight_delta_per_neuron(curr_up, prev_up),
        }


def _relative_norm_change(new_tensor: Tensor, old_tensor: Tensor) -> float:
    """Compute ||new - old|| / ||old||. Handles cross-device tensors."""
    if old_tensor.device != new_tensor.device:
        old_tensor = old_tensor.to(new_tensor.device)
    new_f = new_tensor.float()
    old_f = old_tensor.float()
    delta_norm = (new_f - old_f).norm().item()
    old_norm = old_f.norm().item()
    if old_norm > 1e-10:
        return delta_norm / old_norm
    return 0.0


def _mean_squared_norm_change(new_tensor: Tensor, old_tensor: Tensor) -> Tuple[float, float]:
    """Return (mean(|new - old|²), mean(|old|²)) for deferred ratio computation.

    Basically computes the average l2 norm of vectors, and looks at the ratio of the change to the old l2 norm.
    """
    if old_tensor.device != new_tensor.device:
        old_tensor = old_tensor.to(new_tensor.device)
    new_f = new_tensor.float()
    old_f = old_tensor.float()
    delta_msq = (new_f - old_f).pow(2).sum(-1).mean().item()
    old_msq = old_f.pow(2).sum(-1).mean().item()
    return delta_msq, old_msq


def _rerun_linear(
    linear_module: nn.Module,
    stored_input: Tensor,
) -> Tensor:
    """Re-run a linear layer with stored input and return the output tensor."""
    new_output = linear_module(stored_input)
    if isinstance(new_output, tuple):
        return new_output[0]
    return new_output


def _zero_scalar() -> Tensor:
    """Return a scalar zero tensor on the current CUDA device."""
    return torch.zeros(1, dtype=torch.float32, device=torch.cuda.current_device())


def _get_grad_shard(param: Tensor, param_range) -> Optional[Tensor]:
    """Extract the gradient shard owned by this DP rank.

    Args:
        param: Model parameter with main_grad attribute.
        param_range: Range object with .start and .end for the owned sub-range.

    Returns:
        Float shard tensor, or None if no gradient is available.
    """
    grad = get_param_grad(param)
    if grad is None:
        return None
    return grad.view(-1)[param_range.start:param_range.end].float()


def compute_grad_shard_norm_sq(param: Tensor, param_range) -> Tensor:
    """Compute ||grad_shard||^2 as a scalar CUDA tensor for batched all-reduce.

    Args:
        param: Model parameter with main_grad attribute.
        param_range: Range object with .start and .end for the owned sub-range.

    Returns:
        Scalar CUDA tensor containing ||grad_shard||^2.
    """
    shard = _get_grad_shard(param, param_range)
    if shard is None:
        return _zero_scalar()
    with torch.no_grad():
        return (shard * shard).sum().unsqueeze(0)


def compute_grad_weight_dot_shard(param: Tensor, param_range) -> Tensor:
    """Compute grad_shard . weight_shard as a scalar CUDA tensor for batched all-reduce.

    Args:
        param: Model parameter with main_grad and data attributes.
        param_range: Range object with .start and .end for the owned sub-range.

    Returns:
        Scalar CUDA tensor containing grad_shard . weight_shard.
    """
    shard = _get_grad_shard(param, param_range)
    if shard is None:
        return _zero_scalar()
    with torch.no_grad():
        weight_shard = param.data.view(-1)[param_range.start:param_range.end].float()
        return (shard * weight_shard).sum().unsqueeze(0)
