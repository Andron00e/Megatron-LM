#!/bin/bash
#
# Transformer++ 350M-active MoE on 1 GH200 node (4 GPUs) using ScheduleFree+ (Constant LR).
#
#SBATCH --job-name=transformer-pp-350m-sf_plus-constant
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

# Schedule-Free learning rate is typically set constant.
# Adjust LR as needed.
LR=${LR:-1e-3}
MIN_LR=${MIN_LR:-1e-5}

TP=1
PP=1
CP=1

PROJECT=${WANDB_PROJECT:-megatron-lm-research-baseline}
EXP_NAME=transformer-pp-350m-sf_plus-constant-lr${LR}-$(git rev-parse --short HEAD)
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
    --optimizer schedulefree_plus
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
    # Decoupled weight decay for Schedule-Free+ is scaled by lr**2.
    # Standard values should be much larger (e.g. 20.0 or 50.0).
    --weight-decay 20.0
    --adam-beta1 0.9
    --adam-beta2 0.95
)

SCHEDULEFREE_PLUS_ARGS=(
    --sf-beta1 0.9
    --sf-beta1-max 0.965
    --sf-beta1-anneal-steps 0
    --polyak-beta 0.0
    --c-warmup 0
    --r 0.0
    --weight-lr-power 2.0
)

LEARNING_RATE_ARGS=(
    --lr $LR
    --min-lr $MIN_LR
    --lr-decay-style constant
    --lr-warmup-samples 0
)

INITIALIZATION_ARGS=(
    --seed 42
    --init-method-std 0.02
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

MEGATRON_ARGS=(
    "${TRANSFORMER_ENGINE_ARGS[@]}"
    "${NETWORK_SIZE_ARGS[@]}"
    "${MOE_ARGS[@]}"
    "${TRAINING_ARGS[@]}"
    "${REGULARIZATION_ARGS[@]}"
    "${SCHEDULEFREE_PLUS_ARGS[@]}"
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
