# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""DiLoCo / SNOO / mu^2-DiLoCo: an outer optimizer over the H-step pseudo-gradient.

K workers, each a contiguous slice of the data-parallel group. The inner gradient all-reduce is
confined to a worker's own ranks (DDP is built with ``pg.dp = pg.dp_cp = <worker sub-group>``), so
the K workers' parameters diverge between syncs. Every H iterations every rank computes, over the
*global* DP group,

    Delta_t = x_t - (1/K) sum_k x^(k)_{t,H}

where x_t is the round anchor (``outer_x``), and applies one of the rules below to produce x_{t+1},
which is written back into the fp32 main params (the caller then pushes them to the bf16 model
params). Every rank runs identical arithmetic on the identical all-reduced Delta_t, so all workers
land on the same x_{t+1} by construction and no broadcast is needed.

nesterov / snoo  [douillard2024diloco, kallusky2025snoo]

    b_t = mu b_{t-1} + Delta_t
    x_{t+1} = x_t - lr_out (mu b_t + Delta_t)       (outer_nesterov=True, the DiLoCo/SNOO rule)
    x_{t+1} = x_t - lr_out b_t                      (outer_nesterov=False, heavy-ball)

``snoo`` is exactly the same rule and only additionally asserts workers == 1; it is DiLoCo's
single-worker case, kept as a separate name so arms are self-describing in the logs. mu = 0 and
lr_out = 1 make the outer step a no-op at K = 1 and plain parameter averaging at K > 1.

mu2  [dahan2025mu2sgd], written in the parameterisation of ``megatron/core/optimizer/mu2mars.py``
under the substitution g_t -> Delta_t, with the MARS-approx stale pseudo-gradient
Delta~_t := Delta_{t-1} (option (a) of the plan; the exact re-evaluation needs a second H-step inner
loop, i.e. 2x training compute, and is not implementable here):

    c_t = Delta_t + gamma beta/(1-beta) (Delta_t - Delta_{t-1}),   Delta_0 = 0
    c_t <- c_t * min(1, kappa / ||c_t||_2)      global l2 over all params of this rank, kappa=0 off
    d_t = beta d_{t-1} + (1-beta) c_t,          d_hat_t = d_t / (1 - beta^t)
    v_t = beta2 v_{t-1} + (1-beta2) c_t^2       (precond == 'adam' only)
    P(u) = u                                    (precond == 'sgd')
         = u / (sqrt(v_t / (1 - beta2^t)) + eps)   (precond == 'adam')

    variant 'anytime' (mu^2-SGD option (1); the descent sequence w and the anytime average x):
        w_{t+1} = w_t - lr_out P(d_hat_t)
        x_{t+1} = ga w_{t+1} + (1 - ga) x_t
    variant 'ema' (option (2); one sequence, a second EMA in place of the averaging):
        m_t = beta3 m_{t-1} + (1-beta3) d_hat_t,   m_hat_t = m_t / (1 - beta3^t)
        x_{t+1} = x_t - lr_out P(m_hat_t)

It is x_{t+1}, not w_{t+1}, that starts the workers' next round, so the pseudo-gradient is evaluated
at the average. ga = 1 collapses x == w (the ablation that removes the averaging half); gamma = 0
removes the variance-reduction term; both at once plus beta = 0 reduce 'mu2' to outer SGD.

The bias corrections and the order in which the denominator is applied mirror ``mu2mars.py``,
so that mu2mars and mu2diloco arms are matched on the single number gamma*beta. The objects the
rule acts on are not identical: mu2mars applies the variance-reduction term only to matrix params
(1-D params take a plain AdamW branch) and clips c_t per tensor, while here the pseudo-gradient is
one object -- every param carries the VR term and the clip is the global l2 norm the plan asks for.

The outer state is a set of param-shaped fp32 tensors owned by this controller, keyed off the fp32
main param, and *never* written into the inner ``torch.optim.Optimizer``'s per-param ``state``
dict. Mirroring them there rides the optimizer's own sharding path for free, but it also makes the
inner optimizer's ``state_dict()`` responsible for them, and TransformerEngine's ``FusedAdam`` --
the class ``--optimizer adam`` resolves to inside a TE container -- looks every per-param state key
up in a fixed ``name_to_dtype_map`` and so raises ``KeyError: 'outer_x'`` at the first save (F039).
Instead ``sharded_state_dict()`` emits the outer tensors as ``diloco.<key>.<param>`` ShardedTensors
built from the same model-parameter sharding the optimizer state uses, and the caller stores them
under the checkpoint's own ``diloco`` key; ``load_state_dict()`` adopts them back on resume. Keeping
the state here also keeps it out of the way of every inner optimizer's lazy ``if len(state) == 0``
moment allocation, which an eager write into ``opt.state[p]`` defeated (F020).

The outer clock t is an explicit counter, and a round is H iterations *since the last outer step*
rather than a fixed multiple of H, so a forced (preemption) sync neither replays a t nor shortens
the following round. t itself is not checkpointed -- it only enters the mu2 bias corrections
1-beta^t, saturated after a few rounds -- and is rebuilt on resume as ceil(iteration / H), exact
for a run that was never preempted (the caller asserts ``save_interval % H == 0`` and
``train_iters % H == 0``, and forces an outer step before any unscheduled save).

Transient memory: one fp32 model-sized list of pseudo-gradients is materialised for the duration of
the outer step (the global c_t clip needs two passes over it). Persistent fp32 buffers per param,
above the inner optimizer: 2 for nesterov/snoo, 4 for mu2, 5 with precond='adam'.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional

import torch

_OUTER_OPTIMIZER = None
_LOG_STATS = {}

# Inner optimizer state that is a *copy of the parameters* rather than a moment, and therefore has
# to follow the params through the outer step (see `_rebase_inner_param_state`).
_PARAM_VALUED_INNER_KEYS = ('w',)


@dataclass
class DiLoCoConfig:
    """Outer-loop configuration; field names mirror the ``--diloco-*`` / ``--mu2diloco-*`` flags."""

    workers: int = 1
    inner_steps: int = 0
    outer: str = 'nesterov'
    outer_lr: float = 0.7
    outer_momentum: float = 0.9
    outer_nesterov: bool = True
    outer_precond: str = 'sgd'
    outer_beta2: float = 0.99
    outer_eps: float = 1e-8
    mu2_beta: float = 0.025
    mu2_beta3: float = 0.95
    mu2_gamma: float = 1.0
    mu2_clip: float = 1.0
    mu2_anytime_gamma: float = 0.1
    mu2_variant: str = 'anytime'
    verify_sync: bool = False

    def __post_init__(self):
        if self.outer not in ('nesterov', 'snoo', 'mu2'):
            raise ValueError(f'Invalid --diloco-outer: {self.outer}')
        if self.outer_precond not in ('sgd', 'adam'):
            raise ValueError(f'Invalid --diloco-outer-precond: {self.outer_precond}')
        if self.mu2_variant not in ('anytime', 'ema'):
            raise ValueError(f'Invalid --mu2diloco-variant: {self.mu2_variant}')
        if self.outer == 'snoo' and self.workers > 1:
            raise ValueError('--diloco-outer snoo is the single-worker rule; use nesterov for K>1')
        if not 0.0 <= self.mu2_beta < 1.0:
            raise ValueError(f'Invalid --mu2diloco-beta: {self.mu2_beta}')
        if not 0.0 <= self.outer_beta2 < 1.0:
            raise ValueError(f'Invalid --diloco-outer-beta2: {self.outer_beta2}')
        if not 0.0 <= self.mu2_beta3 < 1.0:
            raise ValueError(f'Invalid --mu2diloco-beta3: {self.mu2_beta3}')
        if not 0.0 < self.mu2_anytime_gamma <= 1.0:
            raise ValueError(f'Invalid --mu2diloco-anytime-gamma: {self.mu2_anytime_gamma}')


def build_worker_process_groups(dp_group, num_workers: int):
    """Split every data-parallel group in the world into `num_workers` contiguous rank slices.

    `torch.distributed.new_group` is collective over the whole world, so the full set of sub-groups
    has to be enumerated in the same order on every rank; the rank lists of all DP groups are
    gathered rather than re-derived from the parallel-state rank arithmetic. Returns this rank's own
    sub-group, which is what DDP should reduce inner gradients over.
    """
    my_ranks = tuple(torch.distributed.get_process_group_ranks(dp_group))
    gathered = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, my_ranks)

    if len(my_ranks) % num_workers != 0:
        raise ValueError(
            f'--diloco-workers {num_workers} does not divide the data-parallel size {len(my_ranks)}'
        )
    slice_size = len(my_ranks) // num_workers

    my_rank = torch.distributed.get_rank()
    my_group = None
    for dp_ranks in sorted(set(gathered)):
        for w in range(num_workers):
            ranks = list(dp_ranks[w * slice_size : (w + 1) * slice_size])
            group = torch.distributed.new_group(ranks=ranks)
            if my_rank in ranks:
                my_group = group
    assert my_group is not None
    return my_group


class DiLoCoOuterOptimizer:
    """The outer loop of DiLoCo / SNOO / mu^2-DiLoCo over an inner ``torch.optim.Optimizer``.

    `optimizers` is the inner torch optimizer (or the list of them, for a ChainedOptimizer) whose
    param groups hold the fp32 main params; `dp_group` is the *global* data-parallel group the
    pseudo-gradient is averaged over, or None for a single-process run.
    """

    def __init__(self, optimizers, config: DiLoCoConfig, dp_group=None):
        if isinstance(optimizers, torch.optim.Optimizer):
            optimizers = [optimizers]
        self.optimizers = list(optimizers)
        self.config = config
        self.dp_group = dp_group
        self.megatron_optimizer = None
        self.outer_state = defaultdict(dict)
        self.completed_iterations = 0
        self.outer_steps = 0
        self.last_sync_iteration = 0
        self.pending = False

    @property
    def items(self) -> List[tuple]:
        """(per-param outer state, param) over every inner optimizer, in a fixed order."""
        return [
            (self.outer_state[p], p)
            for opt in self.optimizers
            for group in opt.param_groups
            for p in group['params']
        ]

    def state_keys(self):
        """Param-shaped fp32 outer state this configuration needs."""
        keys = ['outer_x', 'outer_b']
        if self.config.outer == 'mu2':
            keys.append('outer_dprev')
            keys.append('outer_w' if self.config.mu2_variant == 'anytime' else 'outer_m')
        if self.config.outer_precond == 'adam':
            keys.append('outer_v')
        return keys

    @torch.no_grad()
    def allocate_state(self):
        """Idempotent; `outer_x` and the descent sequence `outer_w` start at the params, not zero.

        The tensors are held here and nowhere else -- see the module docstring on why they must
        not be written into ``opt.state[p]``. A resume allocates first and then overwrites them
        from the checkpoint in ``load_state_dict``.
        """
        keys = self.state_keys()
        for opt in self.optimizers:
            for group in opt.param_groups:
                for p in group['params']:
                    outer = self.outer_state[p]
                    for key in keys:
                        if key in outer:
                            continue
                        if key in ('outer_x', 'outer_w'):
                            outer[key] = p.detach().clone().float()
                        else:
                            outer[key] = torch.zeros_like(p, dtype=torch.float32)

    def attach_to(self, megatron_optimizer):
        """Record the ``MegatronOptimizer`` whose model-param sharding the outer tensors reuse."""
        self.megatron_optimizer = megatron_optimizer

    def _param_to_sharded_param_map(self, model_sharded_state_dict):
        """Map every fp32 main param this controller owns to the model's ShardedTensor for it.

        The same mapping the optimizer builds for its own state: the model params are indexed in
        optimizer order, and ``fp32_from_float16_groups`` is built in lockstep with
        ``float16_groups``, so position i of the two lists is the same weight.
        """
        from megatron.core.dist_checkpointing.optimizer import get_param_id_to_sharded_param_map

        assert self.megatron_optimizer is not None, 'attach_to() was not called'
        param_map = {}
        for opt in getattr(
            self.megatron_optimizer, 'chained_optimizers', [self.megatron_optimizer]
        ):
            if getattr(opt, 'optimizer', None) is None:
                continue
            if getattr(opt, 'float16_groups', None) is not None:
                model_params = [p for group in opt.float16_groups for p in group]
                main_params = [p for group in opt.fp32_from_float16_groups for p in group]
                fp32_params = [p for group in opt.fp32_from_fp32_groups for p in group]
                model_params += fp32_params
                main_params += fp32_params
            else:
                model_params = list(opt.get_parameters())
                main_params = model_params
            id_to_sharded_param = get_param_id_to_sharded_param_map(
                model_sharded_state_dict, model_params
            )
            for index, main_param in enumerate(main_params):
                if index in id_to_sharded_param:
                    param_map[main_param] = id_to_sharded_param[index]
        return param_map

    def sharded_state_dict(self, model_sharded_state_dict, is_loading: bool = False):
        """The outer state as ShardedTensors, for the ``torch_dist`` checkpoint.

        Keys are ``diloco.<outer key>.<model param key>``, i.e. the optimizer-state naming with
        ``diloco`` in place of ``optimizer.state``, and the sharding (including ``replica_id``, so
        one DP rank writes) is the model param's. `is_loading` is accepted for symmetry with
        ``MegatronOptimizer.sharded_state_dict``; the tensors are allocated either way, since a
        load template needs the same buffers a save does.
        """
        from megatron.core.dist_checkpointing.optimizer import make_sharded_optimizer_tensor

        del is_loading
        self.allocate_state()
        param_map = self._param_to_sharded_param_map(model_sharded_state_dict)
        sharded = {}
        for key in self.state_keys():
            entry = {}
            for index, (state, p) in enumerate(self.items):
                if p not in param_map:
                    raise ValueError(f'DiLoCo param {index} does not match any model sharded param')
                entry[index] = make_sharded_optimizer_tensor(
                    param_map[p], state[key], prefix=f'diloco.{key}'
                )
            sharded[key] = entry
        return sharded

    def state_dict(self):
        """The unsharded form, for the plain ``torch`` checkpoint format."""
        self.allocate_state()
        return {key: [state[key] for state, _ in self.items] for key in self.state_keys()}

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        """Adopt a loaded outer state. Indexed by position, so it round-trips both forms above.

        K is not part of it: the outer state is one set of param-shaped tensors whatever the
        worker topology, so a K=1 checkpoint resumes at K=2 and back.
        """
        self.allocate_state()
        items = self.items
        for key in self.state_keys():
            if key not in state_dict:
                raise ValueError(f'DiLoCo state key {key} missing from the checkpoint')
            loaded = state_dict[key]
            if len(loaded) != len(items):
                raise ValueError(
                    f'DiLoCo state {key} has {len(loaded)} tensors, expected {len(items)}'
                )
            for index, (state, _) in enumerate(items):
                state[key].copy_(loaded[index])

    def note_inner_step(self, iteration: int):
        """Record that one inner iteration completed; `iteration` is the count of them.

        Counted whether or not the inner optimizer actually stepped: the grad scaler's skip
        verdict is reduced over the model-parallel group only, so two DiLoCo workers can disagree
        about it, and a round boundary that moved with it would both merge two rounds and hang the
        collective outer step.
        """
        self.completed_iterations = iteration
        self.pending = True

    def should_sync(self, iteration: int) -> bool:
        """A round is H iterations since the last outer step, not a fixed multiple of H.

        So a forced (preemption) sync at an unaligned iteration starts a full-length round instead
        of a short one, and a skipped boundary iteration cannot silently merge two rounds.
        """
        H = self.config.inner_steps
        return H > 0 and iteration - self.last_sync_iteration >= H

    def resume_from(self, iteration: int):
        """Re-anchor the outer clock on a loaded checkpoint's iteration.

        Checkpoints are only written straight after an outer step, so the loaded iteration is a
        round boundary. See the module docstring on why t is rebuilt rather than stored.
        """
        H = self.config.inner_steps
        self.last_sync_iteration = iteration
        self.outer_steps = -(-iteration // H) if H > 0 else 0
        self.pending = False

    @torch.no_grad()
    def outer_step(self, iteration: int):
        """One outer round. `iteration` is the number of completed inner iterations."""
        config = self.config
        t = self.outer_steps + 1
        self.allocate_state()
        items = self.items
        world = 1 if self.dp_group is None else torch.distributed.get_world_size(self.dp_group)

        local_sq = 0.0
        deltas = []
        for state, p in items:
            delta = state['outer_x'].sub(p.detach().float())
            local_sq += delta.square().sum().item()
            if world > 1:
                torch.distributed.all_reduce(delta, group=self.dp_group)
                delta.div_(world)
            deltas.append(delta)

        stats = {
            'diloco/delta-norm': sum(d.square().sum().item() for d in deltas) ** 0.5,
            'diloco/worker-spread': local_sq**0.5,
        }
        if config.outer == 'mu2':
            self._mu2_step(items, deltas, t, stats)
        else:
            self._nesterov_step(items, deltas)
        stats['diloco/outer-update-norm'] = (
            sum(state['outer_b'].square().sum().item() for state, _ in items) ** 0.5
        )

        for state, p in items:
            self._rebase_inner_param_state(p, state['outer_x'])
            p.detach().copy_(state['outer_x'])
        if config.verify_sync and world > 1:
            self._verify_sync([p for _, p in items])

        self.outer_steps = t
        self.last_sync_iteration = iteration
        self.pending = False
        _LOG_STATS.clear()
        _LOG_STATS.update(stats)

    def _nesterov_step(self, items, deltas):
        config = self.config
        for (state, _), delta in zip(items, deltas):
            b = state['outer_b']
            b.mul_(config.outer_momentum).add_(delta)
            if config.outer_nesterov:
                update = b.mul(config.outer_momentum).add_(delta)
            else:
                update = b
            state['outer_x'].add_(update, alpha=-config.outer_lr)

    def _mu2_step(self, items, deltas, t, stats):
        config = self.config
        correction = config.mu2_gamma * config.mu2_beta / (1.0 - config.mu2_beta)

        dot = prev_sq = c_sq = 0.0
        for (state, _), delta in zip(items, deltas):
            dprev = state['outer_dprev']
            dot += delta.mul(dprev).sum().item()
            prev_sq += dprev.square().sum().item()
            c_t = delta.add(delta.sub(dprev), alpha=correction) if correction else delta.clone()
            dprev.copy_(delta)
            c_sq += c_t.square().sum().item()
            delta.copy_(c_t)
        c_norm = c_sq**0.5
        stats['diloco/c-norm'] = c_norm
        stats['diloco/delta-cosine'] = (
            dot / (stats['diloco/delta-norm'] * prev_sq**0.5)
            if prev_sq > 0.0 and stats['diloco/delta-norm'] > 0.0
            else 0.0
        )
        scale = (
            config.mu2_clip / c_norm if config.mu2_clip > 0.0 and c_norm > config.mu2_clip else 1.0
        )
        stats['diloco/clipped'] = float(scale != 1.0)

        beta = config.mu2_beta
        bias1 = 1.0 - beta**t
        bias2 = 1.0 - config.outer_beta2**t
        bias3 = 1.0 - config.mu2_beta3**t
        for (state, _), c_t in zip(items, deltas):
            if scale != 1.0:
                c_t.mul_(scale)
            d = state['outer_b']
            d.mul_(beta).add_(c_t, alpha=1.0 - beta)
            if config.outer_precond == 'adam':
                v = state['outer_v']
                v.mul_(config.outer_beta2).addcmul_(c_t, c_t, value=1.0 - config.outer_beta2)
                denom = v.sqrt().mul_(1.0 / bias2**0.5).add_(config.outer_eps)
            else:
                denom = None
            if config.mu2_variant == 'anytime':
                update = d.div(bias1)
                if denom is not None:
                    update = update.div_(denom)
                w = state['outer_w']
                w.add_(update, alpha=-config.outer_lr)
                state['outer_x'].mul_(1.0 - config.mu2_anytime_gamma).add_(
                    w, alpha=config.mu2_anytime_gamma
                )
            else:
                m = state['outer_m']
                m.mul_(config.mu2_beta3).add_(d.div(bias1), alpha=1.0 - config.mu2_beta3)
                update = m.div(bias3)
                if denom is not None:
                    update = update.div_(denom)
                state['outer_x'].add_(update, alpha=-config.outer_lr)

    def _rebase_inner_param_state(self, param, outer_x):
        """Translate inner state that is a copy of the params, not a moment, by the outer step.

        Called with the pre-jump `param` still in place, once per param, so the displacement
        x_{t+1} - x^(k)_{t,H} is available without a second copy of the model.

        Keeping the inner *moments* across an outer step is what DiLoCo prescribes. A
        parameter-valued inner state is different: mu2mars's anytime descent sequence w sits on
        the pre-jump trajectory, and its next inner step pulls p back toward it by anytime_gamma
        of the whole outer update, partially undoing the outer step. Setting w := x_{t+1} removes
        that drag, but the anytime pair's steady state is a *lead* w - p = -((1-ga)/ga) lr u, and
        it is that lead, not ga, that makes p advance by the full lr u per inner step: a reset
        would run the first ~2/ga iterations of every round at a reduced effective inner LR (22
        iterations at the default ga = 0.1, against H = 30-100), mismatching the inner step of a
        mu2diloco arm against its non-DiLoCo mu2mars control by an amount that depends on H.
        Translating w by the same displacement the params take removes the drag and preserves the
        lead: the inner optimizer sees only its new position.
        """
        for opt in self.optimizers:
            state = opt.state.get(param)
            if not state:
                continue
            for key in _PARAM_VALUED_INNER_KEYS:
                if key in state:
                    state[key].sub_(param.detach()).add_(outer_x)

    def _verify_sync(self, params):
        """Max-abs spread per parameter: a signed whole-model sum hides cancelling divergence."""
        group = self.dp_group
        gap = 0.0
        for p in params:
            hi = p.detach().float().clone()
            lo = hi.clone()
            torch.distributed.all_reduce(hi, op=torch.distributed.ReduceOp.MAX, group=group)
            torch.distributed.all_reduce(lo, op=torch.distributed.ReduceOp.MIN, group=group)
            gap = max(gap, hi.sub_(lo).max().item())
        assert gap == 0.0, f'DiLoCo workers disagree after the outer step: max spread {gap}'


def set_diloco_outer_optimizer(outer_optimizer: Optional[DiLoCoOuterOptimizer]):
    """Module-level handle, so the hook and the checkpoint paths need no signature changes."""
    global _OUTER_OPTIMIZER
    _OUTER_OPTIMIZER = outer_optimizer


def get_diloco_outer_optimizer() -> Optional[DiLoCoOuterOptimizer]:
    return _OUTER_OPTIMIZER


def get_diloco_log_stats() -> dict:
    """Consume-and-clear, so an outer round is logged once, at the iteration it happened."""
    stats = dict(_LOG_STATS)
    _LOG_STATS.clear()
    return stats
