# Model Internals Logging - Metric Definitions

This document provides precise mathematical definitions for all metrics captured by the internals logging system.

**Notation.** We write $G := \nabla_\theta \mathcal{L}$ for the gradient of the loss with respect to a parameter $\theta$.

**Key format.** Per-parameter metrics use `/`-separated paths derived from `named_parameters()`. The `layers/<N>/` segment is moved to the end as a zero-padded `layer_NN` suffix so that metrics sort lexicographically by layer on dashboards.

For example, a parameter named `decoder.layers.0.self_attention.linear_qkv.weight` is logged as `decoder/self_attention/linear_qkv/weight/layer_00` (with the appropriate metric prefix).

---

## 1. Activation Statistics

**Logged as:** `activations/<metric>/layer_{N:02d}`

| Metric | Definition |
|--------|------------|
| `mean` | $\mu = \frac{1}{n}\sum_{i=1}^{n} x_i$ |
| `var` | $\sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \mu)^2$ |
| `rms` | $\text{RMS} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2}$ |
| `min` | $\frac{1}{N}\sum_{j=1}^{N} \min_k \; v_{j,k}$ &nbsp;&nbsp; *(expected per-vector minimum)* |
| `max` | $\frac{1}{N}\sum_{j=1}^{N} \max_k \; v_{j,k}$ &nbsp;&nbsp; *(expected per-vector maximum)* |
| `kurtosis` | $\kappa = \frac{E_k[s_k^2]}{E_k[s_k]^2}$ where $s_k = \frac{1}{N}\sum_j x_{j,k}^2$ &nbsp;&nbsp; *(per-neuron energy kurtosis; 1 = uniform, >1 = non-uniform)* |

Where $X$ is the activation tensor reshaped to $[N, d]$ (N vectors of dimension d) from the layer output. Per-vector stats (min, max, rms) are computed along the last dimension and then averaged over vectors. All stats are AVG-reduced across DP ranks.

---

## 2. Gradient Statistics

### Per-Parameter Norms

**Logged as:** `gradients/norm/<param_path>/layer_NN`

$$\|G\|_2 = \sqrt{\sum_i G_i^2}$$

### Per-Layer Aggregates

**Logged as:** `per_layer_gradients/<metric>/layer_{N:02d}`

| Metric | Definition |
|--------|------------|
| `total_norm` | $\sqrt{\sum_{j \in \text{layer}_N} \lVert G_j\rVert_2^2}$ |
| `avg_norm` | $\frac{1}{m}\sum_{j \in \text{layer}_N} \lVert G_j\rVert_2$ |
| `max_norm` | $\max_{j \in \text{layer}_N} \lVert G_j\rVert_2$ |

### Gradient Flow (Between Layers)

**Logged as:** `gradient_flow/layer_{N:02d}_to_{N+1:02d}`

$$\text{flow}_{N \to N+1} = \frac{\text{total norm}_{N+1}}{\text{total norm}_N}$$

Values $< 1$ indicate vanishing gradients; values $> 1$ indicate exploding gradients.

---

## 3. Relative Weight Updates (δW)

### Per-Parameter

**Logged as:** `delta_W/<param_path>/layer_NN`

$$\delta_W = \frac{\|W_t - W_{t-1}\|_2}{\|W_{t-1}\|_2}$$

### Per-Neuron Statistics

**Logged as:** `delta_W_per_neuron/mean/<param_path>/layer_NN`

For a weight matrix $W \in \mathbb{R}^{m \times n}$, compute per-row (per-neuron) relative changes:

$$\delta_{W,j} = \frac{\|W_{t,j} - W_{t-1,j}\|_2}{\|W_{t-1,j}\|_2} \quad \text{for row } j$$

| Metric | Definition |
|--------|------------|
| `mean` | $\frac{1}{m}\sum_{j=1}^{m} \delta_{W,j}$ |

---

## 4. Angular Metrics

These metrics measure directional changes in weight space, particularly useful for spherical/normalized training methods (e.g., nGPT, EDM2).

### Per-Parameter

**Logged as:** `angular/<metric>/<param_path>/layer_NN`

| Metric | Definition |
|--------|------------|
| `cos_similarity` | $\cos(\theta) = \frac{W_t \cdot W_{t-1}}{\lVert W_t\rVert_2 \cdot \lVert W_{t-1}\rVert_2}$ |
| `degrees` | $\theta = \arccos(\cos(\theta)) \times \frac{180}{\pi}$ |

**Interpretation:**
- $\cos(\theta) \approx 1$: weight direction unchanged
- $\cos(\theta) \approx 0$: orthogonal (90° change)
- $\cos(\theta) \approx -1$: opposite direction (180° change)

### Per-Layer Aggregates

**Logged as:** `per_layer_angular/<metric>/layer_{N:02d}`

| Metric | Definition |
|--------|------------|
| `degrees` | Average angular change across all parameters in layer |
| `max_degrees` | Maximum angular change in layer |
| `cos_similarity` | Average cosine similarity across layer |

---

## 5. Gradient-Weight Alignment

These metrics decompose the gradient into components parallel and perpendicular to the current weight vector.

### Per-Parameter

**Logged as:** `grad_weight_align/<metric>/<param_path>/layer_NN`

| Metric | Definition |
|--------|------------|
| `cos` | $\cos(G, W) = \frac{G \cdot W}{\lVert G\rVert_2 \cdot \lVert W\rVert_2}$ |
| `radial` | $r = \lVert G\rVert_2 \cdot \cos(G, W)$ |
| `tangential` | $t = \lVert G\rVert_2 \cdot \sqrt{1 - \cos^2(G, W)}$ |

### Geometric Interpretation

```
                    G (gradient)
                   /|
                  / |
                 /  | tangential (t)
                /   |
               /    |
              /_____|___________ W (weight direction)
                radial (r)
```

**For spherical/normalized training:**
- **Tangential component**: Updates that change weight *direction* while preserving norm
- **Radial component**: Updates that change weight *magnitude*
- High tangential + low radial → norm-preserving directional updates

### Per-Layer Aggregates

**Logged as:** `per_layer_grad_weight_align/<metric>/layer_{N:02d}`

| Metric | Definition |
|--------|------------|
| `cos` | Average cosine alignment across layer parameters |
| `radial` | Average radial component across layer |
| `tangential` | Average tangential component across layer |

---

## 6. Parameter Norms (from `training.py`)

**Logged as:** `params-norm/<param_path>/layer_NN`

$$\|\theta\|_2 = \sqrt{\sum_i \theta_i^2}$$

Computed with proper synchronization across data-parallel groups (unsure about tensor-parallel and pipeline-parallel).

---

## Summary

| Category | Metrics per Layer | W&B Prefix |
|----------|-------------------|------------|
| Activation Statistics | 6 | `activations/<metric>/layer_NN` |
| Gradient Statistics (per-param) | per-param | `gradients/norm/` |
| Gradient Statistics (per-layer) | 3 | `per_layer_gradients/` |
| Gradient Flow | 1 (per layer pair) | `gradient_flow/` |
| Weight Updates (per-param) | per-param | `delta_W/`, `delta_W_per_neuron/mean/` |
| Angular Metrics (per-param) | per-param | `angular/` |
| Angular Metrics (per-layer) | 3 | `per_layer_angular/` |
| Gradient-Weight Alignment (per-param) | per-param | `grad_weight_align/` |
| Gradient-Weight Alignment (per-layer) | 3 | `per_layer_grad_weight_align/` |
| Parameter Norms (per-param) | per-param | `params-norm/` |

---

## Performance Notes

### GPU Memory vs Transfer Overhead

By default, previous weights are stored on CPU to conserve GPU memory. This causes GPU↔CPU transfers on logging iterations, which can introduce throughput variance.

Use `--internals-weights-on-gpu` to keep previous weights on GPU:
- **Pros**: Eliminates PCIe transfer overhead and throughput variance
- **Cons**: Uses additional GPU memory (~1x model parameter size)

Recommended when:
- You have sufficient GPU memory headroom
- Consistent iteration timing is important for profiling
- Using `--log-relative-updates` or `--log-angular-metrics`
