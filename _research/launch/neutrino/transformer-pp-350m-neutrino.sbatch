#!/bin/bash
#
# Transformer++ 350M-active MoE on 1 GH200 node (4 GPUs).
# 24L / 1024H / 16h / 4kv (GQA). MoE: 16 routed experts top-1 + 1 shared,
# ffn-hidden 1280 (iso-active to dense 2560). QK-LayerNorm. FP8 blockwise.
# SwiGLU, RMSNorm, RoPE, Neutrino optimizer, WSD.
# 15B tokens on ClimbMix, ~57220 steps at 262K tokens/step (MBS=8).
#
# Architecture matches the 350m-wsdmc MoE runs for apples-to-apples comparison.
#
# Neutrino =Low-Rank Polar Updates with Shared-Seed Basis
# and Local Error Feedback. Embedding + LM head + Biases/norms use fallback Adam.
#
# --overlap-param-gather intentionally omitted (breaks Muon-style orthogonal updates).
#
#SBATCH --job-name=transformer-pp-350m-neutrino
#SBATCH --time=10:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/megatron-lm-research-baseline/_research/results/runs/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/megatron-lm-research-baseline/_research/results/runs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --no-requeue

set -euo pipefail
: "${MEGATRON_DATA_PATH:?please set MEGATRON_DATA_PATH to the Megatron-binary dataset prefix (without .bin/.idx), e.g. /path/to/climbmix_small}"
echo "START TIME: $(date)"
export APERTUS_PHASE_SBATCH_START=$(date +%s.%N)

REPO_DIR=/iopsstor/scratch/cscs/$USER/megatron-lm-research-baseline
WORKDIR=${SLURM_SUBMIT_DIR:-$REPO_DIR}
cd $WORKDIR

PACKAGE_DIR=$REPO_DIR/_research/packages
DATASET_CACHE_DIR=$REPO_DIR/cache/dataset
TMP_CACHE_DIR=$REPO_DIR/cache/tmp

export PYTHONPATH=$WORKDIR:$PACKAGE_DIR:$WORKDIR${PYTHONPATH:+:$PYTHONPATH}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_CPP_LOG_LEVEL=WARNING
export TRITON_CACHE_DIR=$REPO_DIR/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$REPO_DIR/cache/inductor
export TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TMPDIR=$TMP_CACHE_DIR

mkdir -p $DATASET_CACHE_DIR $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $TMP_CACHE_DIR $PACKAGE_DIR _research/results/runs _research/results/performance _research/results/ckpts

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
MASTER_PORT=25678

NUM_LAYERS=24
HIDDEN=1024
FFN_HIDDEN=2560
NUM_HEADS=16
NUM_KV_HEADS=4

MBS=8
GBS=128
SEQ_LEN=4096
TRAIN_SAMPLES=3662109

# LR knobs. For Neutrino, MLR (matrix learning rate) controls the matrix updates,
# and LR controls the fallback Adam parameters (embeddings, biases, norms).
# Override via env: MLR=8e-3 sbatch ... or LR=1e-3 sbatch ...
LR=${LR:-1e-3}
MLR=${MLR:-8e-3}
MIN_LR=${MIN_LR:-1e-5}

# Neutrino options
NEUTRINO_K=${NEUTRINO_K:-512}
NEUTRINO_BASIS_INIT=${NEUTRINO_BASIS_INIT:-gaussian}  # Choices: gaussian, rademacher, uniform, orthonormal
NEUTRINO_NO_MOMENTUM=${NEUTRINO_NO_MOMENTUM:-false}    # Set to true to disable momentum buffer VRAM allocation

TP=1
PP=1
CP=1

PROJECT=${WANDB_PROJECT:-megatron-lm-research-baseline}
EXP_NAME=transformer-pp-350m-neutrino-lr${LR}-mlr${MLR}-init-${NEUTRINO_BASIS_INIT}-$(git rev-parse --short HEAD)
CKPT_DIR=${MASTER_CKPT_DIR:-$WORKDIR/_research/results/ckpts/$EXP_NAME}
mkdir -p $CKPT_DIR

export APERTUS_LOG_DIR=$WORKDIR/_research/results/performance
export APERTUS_FEATURE=transformer-pp
export APERTUS_TRACK=transformer-pp-350m
export APERTUS_RUN_NAME=$EXP_NAME-$SLURM_JOB_ID
export APERTUS_LOG_ACT_STATS=1
export APERTUS_LOG_TOP1_ACC=1
export APERTUS_LOG_ROW_CV=1
export APERTUS_LOG_NEURON_STATS=1
export APERTUS_LOG_NEURON_INTERVAL=50
export APERTUS_LOG_PER_LAYER_GRADS=1
export APERTUS_LOG_LOSS_SPIKES=1

TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
    --main-grads-dtype fp32
)

NETWORK_SIZE_ARGS=(
    --num-layers $NUM_LAYERS
    --hidden-size $HIDDEN
    --ffn-hidden-size $FFN_HIDDEN
    --num-attention-heads $NUM_HEADS
    --group-query-attention
    --num-query-groups $NUM_KV_HEADS
    --max-position-embeddings $SEQ_LEN
    --position-embedding-type rope
    --rotary-base 500000
    --make-vocab-size-divisible-by 128
    --normalization RMSNorm
    --swiglu
    --qk-layernorm
    --untie-embeddings-and-output-weights
    --attention-backend auto
    --seq-length $SEQ_LEN
)

MOE_ARGS=(
    --num-experts 16
    --moe-router-topk 1
    --moe-router-pre-softmax
    --moe-ffn-hidden-size 1280
    --moe-shared-expert-intermediate-size 1280
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-z-loss-coeff 1e-3
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-token-dispatcher-type alltoall
    --moe-per-layer-logging
)

TRAINING_ARGS=(
    --micro-batch-size $MBS
    --global-batch-size $GBS
    --train-samples $TRAIN_SAMPLES
    --exit-duration-in-mins 595
    --log-interval 1
    --eval-interval $TRAIN_SAMPLES
    --eval-iters 0
    --optimizer neutrino
    --dataloader-type single
    --manual-gc
    --manual-gc-interval 50
    --cross-entropy-loss-fusion
    --disable-bias-linear
    --no-check-for-nan-in-loss-and-grad
)

REGULARIZATION_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --weight-decay 0.0
    --adam-beta1 0.9
    --adam-beta2 0.99
)

# Neutrino configuration parameters
NEUTRINO_ARGS=(
    --neutrino-k $NEUTRINO_K
    --neutrino-basis-init $NEUTRINO_BASIS_INIT
    --muon-momentum 0.95
    --muon-nesterov
    --muon-scale-mode shape_up
    --muon-tp-mode duplicated
    --muon-scalar-lr $LR
    --muon-scalar-weight-decay 0.0
)

if [ "$NEUTRINO_NO_MOMENTUM" = "true" ]; then
    NEUTRINO_ARGS+=( --neutrino-no-momentum )
fi

LEARNING_RATE_ARGS=(
    --lr $MLR
    --min-lr $MIN_LR
    --lr-decay-style WSD
    --lr-warmup-samples 0
    --lr-wsd-decay-style minus_sqrt
    --lr-wsd-decay-samples 732422
)

INITIALIZATION_ARGS=(
    --seed 42
    --init-method-std 0.02
    # Fixed scalar multiplied into embeddings in forward (HIDDEN=1024 -> sqrt(1024)=32).
    --embedding-multiplier 32.0
)

MIXED_PRECISION_ARGS=(
    --bf16
    --fp8-format e4m3
    --fp8-recipe blockwise
)

DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size $TP
    --pipeline-model-parallel-size $PP
    --context-parallel-size $CP
    --expert-model-parallel-size 1
    --use-distributed-optimizer
    --overlap-grad-reduce
    --sequence-parallel
)

LOGGING_ARGS=(
    --save $CKPT_DIR
    --load $CKPT_DIR
    --save-interval 5722
    --ckpt-format torch
    --log-throughput
    --log-progress
    --log-memory-to-tensorboard
    --log-params-norm
    --tensorboard-dir $WORKDIR/_research/results/runs/tb-$SLURM_JOB_ID
    --tensorboard-log-interval 1
    --log-timers-to-tensorboard
)

TOKENIZER_ARGS=(
    --tokenizer-type GPT2BPETokenizer
    --vocab-file $WORKDIR/_research/data/gpt2-vocab.json
    --merge-file $WORKDIR/_research/data/gpt2-merges.txt
)

DATA_ARGS=(
    --data-path $MEGATRON_DATA_PATH
    --data-cache-path $DATASET_CACHE_DIR
    --split 99,1,0
    --num-workers 4
)

# Flat array so `interactive-run.sh` (which expands "${MEGATRON_ARGS[@]}")
# sees each token as its own argv element. Concatenating into a string and
# referencing it as ${VAR[@]} collapses to a single argv element under
# argparse, which fails with "unrecognized arguments: <everything>".
MEGATRON_ARGS=(
    "${TRANSFORMER_ENGINE_ARGS[@]}"
    "${NETWORK_SIZE_ARGS[@]}"
    "${MOE_ARGS[@]}"
    "${TRAINING_ARGS[@]}"
    "${REGULARIZATION_ARGS[@]}"
    "${NEUTRINO_ARGS[@]}"
    "${LEARNING_RATE_ARGS[@]}"
    "${INITIALIZATION_ARGS[@]}"
    "${MIXED_PRECISION_ARGS[@]}"
    "${DISTRIBUTED_ARGS[@]}"
    "${LOGGING_ARGS[@]}"
    "${TOKENIZER_ARGS[@]}"
    "${DATA_ARGS[@]}"
)

if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_PROJECT=$PROJECT
    export WANDB_NAME=$EXP_NAME-$SLURM_JOB_ID
    MEGATRON_ARGS+=(
        --wandb-project "$PROJECT"
        --wandb-exp-name "$EXP_NAME-$SLURM_JOB_ID"
        --wandb-save-dir "$WORKDIR/_research/results/runs"
    )
    if [ -n "${WANDB_ENTITY:-}" ]; then
        MEGATRON_ARGS+=( --wandb-entity "$WANDB_ENTITY" )
    fi
else
    export WANDB_MODE=disabled
fi

MEGATRON_ARGS_STR="${MEGATRON_ARGS[@]}"

ENV_SETUP="export MASTER_ADDR=$MASTER_ADDR; \
    export MASTER_PORT=$MASTER_PORT; \
    export RANK=\$SLURM_PROCID; \
    export WORLD_SIZE=\$SLURM_NTASKS; \
    export LOCAL_RANK=\$SLURM_LOCALID; \
    export LOCAL_WORLD_SIZE=$((SLURM_NTASKS/SLURM_NNODES)); \
    export PYTHONPATH=$WORKDIR:$PACKAGE_DIR:$WORKDIR\${PYTHONPATH:+:\$PYTHONPATH}; \
    cd $WORKDIR"

TRAINING_CMD="bash $WORKDIR/_research/launch/install_python_deps.sh $PACKAGE_DIR && python3 $WORKDIR/pretrain_gpt.py $MEGATRON_ARGS_STR"

echo "CMD: $TRAINING_CMD"
export APERTUS_PHASE_PRE_SRUN=$(date +%s.%N)
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=$WORKDIR/_research/launch/alps3.toml \
    --cpus-per-task $SLURM_CPUS_PER_TASK --wait 120 \
    bash -c "export APERTUS_PHASE_CONTAINER_STARTED=\$(date +%s.%N); $ENV_SETUP; $TRAINING_CMD"

echo "END TIME: $(date)"
