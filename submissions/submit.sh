# TODO: fp8dpa
# TODO: extra logs

#= Prelude =#
# General settings.
#CONTAINER=/iopsstor/scratch/cscs/ahernnde/ngc_25-12-alps1.toml
CONTAINER=/iopsstor/scratch/cscs/ahernnde/ncg_new_v2.toml

SCRIPT_VERSION=v1
SEQ_LEN=4096
TOKENIZER=mistralai/Mistral-Nemo-Base-2407
TOKENIZED_DATA_PATH=/iopsstor/scratch/cscs/jpcoles/a06/swissai-fineweb-edu-filterrobots-merge
DATA_PATHS=$(find $TOKENIZED_DATA_PATH -type f | sed -E 's/\.[^.]+$//' | sort -u | tr '\n' ' ')
CODE_PATH=$PWD
EMERGING_OPTIMIZERS_PATH=/iopsstor/scratch/cscs/ahernnde/opts/Emerging-Optimizers
TRAIN_ROOT=$SCRATCH/opts/logs

# tmps.
TRITON_HOME_DIR=/tmp/.triton
TORCH_INDUCTOR_CACHE_DIR=/tmp/.torch_inductor
PYTHON_CACHE_DIR=/tmp/.python_cache

# Hardware defaults.
NODES=1
TIME=12:00:00

# Arch defaults.
NORMALIZATION=RMSNorm

# Opt defaults.
WEIGHT_DECAY=0.1
WEIGHT_DECAY_METHOD=decoupled
MIN_LR=1e-8
OPT=adam

BETA1=0.9
BETA2=0.99
BETA3=0.999
MUON_MOMENTUM=0.95
ADAMBETA1=0.95
ALPHA=5
MUON_SCALE_MODE=spectral

HYPERBALL=false
HS_KIND=l2
HS_R=1
HS_UPDATE=false
HS_EMBED=false
HS_SPLIT_HEADS=false

ACTIVATION=swiglu

NO_WARMUP=false
DECAY=wsd
COOLDOWN=0.2

# Misc. defaults.
EXTRA_LOG=true
NO_SAVE=false

# Usage function.
usage () {
	echo "Usage: submit.sh <size> [options...]"
	echo "<size>: 110/390/1"
	echo "Options:"
	# Misc settings.
	echo " --nodes <nodes>: How many nodes to use"
	echo " --debug: Enable debug mode"
	echo " --extra-name <name>: Add a suffix to the name"
	echo " --time (default=$TIME): Change the sbatch time limit"
	echo " --no-extra-logs: Disable costly logging"
	echo " --no-save: Disable saving checkpoint"
	# FP8 settings.
	echo " --fp8: Enables fp8"
	# Training settings..
	echo " --tokens <int>: Amount of tokens to train with (in B)."
	echo " --lr <float>: Learning rate."
	echo " --no-warmup: Deactivates learning rate warmup"
	echo " --decay <wsd/cos>"
	echo " --cooldown <float>: Fraction to do cooldown"
	# Architecture settings.
	echo " --init <float>: Change init std."
	echo " --activation (default=$ACTIVATION): MLP activation. Choices=[swiglu, gelu]."
	echo " --no-pre-norm"
	echo " --no-final-layernorm"
	echo " --normalization <RMSNorm/L2Norm>"
	echo " --no-learnable-norms"
	echo " --post-norm"
	echo " --post-block-norm"
	echo " --use-stream-minus-residual"
	echo " --layer-scale <float>"
	echo " --layer-scale-scale <float>"
	echo " --residual-layer-scale <float>"
	echo " --residual-layer-scale-scale <float>"
	echo " --softmax-scale <float>"
	echo " --qk-norm <RMSNorm/L2Norm>"
	echo " --mlp-layer-scale <float>"
	echo " --mlp-layer-scale-gate-scale <float>"
	echo " --mlp-out-scale <float>"
	echo " --logits-layer-scale <float>"
	echo " --logits-layer-scale-scale <float>"
	echo " --upscale-embedding <float>"
	# Optimizer settings.
	echo " --opt <adam/dmuon/muon/dmaster/master/ademamix> (default=$OPT)"
	echo " --master-orthogonalize"
	echo " --b1: beta1 (master&adam&muon&ademamix)"
	echo " --b2: beta2 (master&adam&ademamix)"
	echo " --mb1: beta1 (master&muon)"
	echo " --b3: beta3 (master&ademamix)"
	echo " --alpha: ademamix alpha"
	echo " --muon-scale <spectral/shape_scaling/unit_rms_norm/none>"
	echo " --muon-nesterov: Enables muon nesterov momentum"
	echo " --mlr: muon learning rate"
	echo " --wd: weight decay"
	echo " --wd-method (decoupled/independent): weight decay method"
	echo " --hs <row/col/rowcol/flat>: Enables hypersphere training"
	echo " --hs-kind <l2/standard/spectral>: hypersphere kind"
	echo " --hs-r <learnable/float>: hypersphere radius"
	echo " --hs-u: hypersphere normalize update"
	echo " --hs-embed: hypersphere normalize embeddings"
	echo " --hs-emb-no-orthogonal: Don't use muon update on embeddings, but keep fixed to the sphere (if --hs-embed is also set, otherwise no effect)"
	echo " --hs-split-heads: hypersphere normalize q,k,v heads separately"
	echo " --hs-p: project gradient to tangent space"
	echo " --hs-s: soft hyperball norm clipping."
	# Logs.
	echo " --wandb-name <str>: Specify wandb name."
}

if [[ $# -eq 0 ]]; then
	>&2 echo Invalid argument count: $#
	usage
	exit 1
fi

# Define some variables depending on size.
EXTRA_ARGS=()
SCALE=B  # M or B.
TP=1
PP=1
UNTIE=true
INIT_STD=0.02
if [[ $1 -eq 110 ]]; then 
	# batch_size: ~0.52M.
	LAYERS=12
	HIDDEN_SIZE=512
	FFN_SIZE=2048
	NUM_HEADS=4
	NUM_QUERY_GROUPS=2
	MBS="${MBS:-8}"
	GBS=128
	ITERS_PER_BT=2000
	LR=0.002
	SIZE=110
	SAVE_FREQ=10000
	DEF_TOKENS=25
	INTERMEDIATE_METRICS_INTERVAL=10
	SCALE=M
	UNTIE=false
	LOG_FREQ=50
elif [[ $1 -eq 390 ]]; then 
	# batch_size: ~0.52M.
	LAYERS=16
	HIDDEN_SIZE=1024
	FFN_SIZE=4096
	NUM_HEADS=8
	NUM_QUERY_GROUPS=4
	MBS="${MBS:-8}"
	GBS=128
	ITERS_PER_BT=2000
	LR=0.001
	SIZE=390
	SAVE_FREQ=10000
	DEF_TOKENS=50
	INTERMEDIATE_METRICS_INTERVAL=10
	SCALE=M
	UNTIE=false
	LOG_FREQ=100
elif [[ $1 -eq 1 ]]; then 
	# batch_size: ~1.05M.
	LAYERS=16
	HIDDEN_SIZE=2048
	FFN_SIZE=8192
	NUM_HEADS=16
	NUM_QUERY_GROUPS=8
	MBS="${MBS:-4}"
	GBS=256
	ITERS_PER_BT=1000
	LR=0.00025
	SIZE=1.5
	SAVE_FREQ=5000
	DEF_TOKENS=125
	INTERMEDIATE_METRICS_INTERVAL=100
	LOG_FREQ=200
else
	>&2 echo "Invalid model size: $1"
	usage
	exit 1
fi
shift

# Now get the general options.
TOKENS=$DEF_TOKENS
SUFFIX=""
while [[ $# -gt 0 ]]; do
	case $1 in
		# Misc settings.
		--nodes) NODES=$2; shift 2;;
		--debug)
			SCRIPT_VERSION=$SCRIPT_VERSION-debug; shift;;
		--extra-name)
			EXTRA_NAME="-$2"; shift 2;;
		--time)
			TIME=$2; shift 2;;
		--no-extra-logs)
			EXTRA_LOG=false; shift;;
		--no-save)
			NO_SAVE=true; shift;;
		# FP8 settings.
		--fp8)
			FP8=true; shift;;
		# Training settings.
		--tokens)
			TOKENS=$2; shift 2;;
		--lr)
			LR=$2; 
			CHANGED_LR=true
			shift 2;;
		--no-warmup)
			NO_WARMUP=true; shift;;
		--decay)
			DECAY=$2; shift 2;;
		--cooldown)
			COOLDOWN=$2; shift 2;;
		# Architecture settings.
		--init)
			NEW_INIT_STD=$2; shift 2;;
		--activation)
			ACTIVATION=$2; shift 2;;
		--no-pre-norm)
			NO_PRE_NORM=true; shift;;
		--no-final-layernorm)
			NO_FINAL_LAYERNORM=true; shift;;
		--normalization)
			NORMALIZATION=$2; shift 2;;
		--no-learnable-norms)
			NO_LEARNABLE_NORMS=true; shift;;
		--post-norm)
			POST_NORM=true; shift;;
		--post-block-norm)
			POST_BLOCK_NORM=true; shift;;
		--use-stream-minus-residual)
			USE_STREAM_MINUS_RESIDUAL=true; shift;;
		--layer-scale)
			LAYER_SCALE=$2; shift 2;;
		--layer-scale-scale)
			LAYER_SCALE_SCALE=$2; shift 2;;
		--residual-layer-scale)
			RESIDUAL_LAYER_SCALE=$2; shift 2;;
		--residual-layer-scale-scale)
			RESIDUAL_LAYER_SCALE_SCALE=$2; shift 2;;
		--softmax-scale)
			SOFT_MAX_SCALE=$2; shift 2;;
		--qk-norm)
			QK_NORM=$2; shift 2;;
		--qk-layer-scale)
			QK_LAYER_SCALE=$2; shift 2;;
		--qk-layer-scale-scale)
			QK_LAYER_SCALE_SCALE=$2; shift 2;;
		--mlp-layer-scale)
			MLP_LAYER_SCALE=$2; shift 2;;
		--mlp-layer-scale-gate-scale)
			MLP_LAYER_SCALE_GATE_SCALE=$2; shift 2;;
		--mlp-out-scale)
			MLP_OUT_SCALE=$2; shift 2;;
		--logits-layer-scale)
			LOGITS_LAYER_SCALE=$2; shift 2;;
		--logits-layer-scale-scale)
			LOGITS_LAYER_SCALE_SCALE=$2; shift 2;;
		--upscale-embedding)
			UPSCALE_EMBEDDING=$2; shift 2;;
		# Opt settings.
		--opt)
			OPT=$2; shift 2;;
		--master-orthogonalize)
			MASTER_ORTHOGONALIZE=true; shift;;
		--b1)
			BETA1=$2; shift 2;;
		--b2)
			BETA2=$2; shift 2;;
		--b3)
			BETA3=$2; shift 2;;
		--mb1)
			ADAMBETA1=$2; shift 2;;
		--mlr)
			MUON_LR_FACTOR=$2; shift 2;;
		--alpha)
			ALPHA=$2; shift 2;;
		--muon-scale)
			MUON_SCALE_MODE=$2; shift 2;;
		--muon-nesterov)
			MUON_NESTEROV=true; shift;;
		--wd)
			WEIGHT_DECAY=$2; shift 2;;
		--wd-method)
			WEIGHT_DECAY_METHOD=$2; shift 2;;
		--hs)
			HYPERBALL=$2; shift 2;;
		--hs-kind)
			HS_KIND=$2; shift 2;;
		--hs-r)
			HS_R=$2; shift 2;;
		--hs-u)
			HS_UPDATE=true; shift;;
		--hs-embed)
			HS_EMBED=true; shift;;
		--hs-embed-no-orthogonal)
			HS_EMBED_NO_ORTHOGONAL=true; shift;;
		--hs-split-heads)
			HS_SPLIT_HEADS=true; shift;;
		--hs-p)
			HS_PROJECT=true; shift;;
		--hs-s)
			HS_SOFT=true; shift;;
		# Logs.
		--wandb-name)
			WANDB_NAME=$2; shift 2;;
		*)
			echo "Unexpected argument $1"
			usage
			exit 1
	esac
done

#= MIDDLE: Set up arguments. =#
# Opt settings.
OPT_ARGS=()
if [[ $OPT = adam ]]; then
	OPT_ARGS+=(--overlap-grad-reduce --use-distributed-optimizer)
	if [[ $BETA1 != 0.9 ]] || [[ $BETA2 != 0.99 ]]; then
		SUFFIX=${SUFFIX}-b${BETA1}_$BETA2
	fi
elif [[ $OPT = muon ]] || [[ $OPT = dmuon ]]; then
	SUFFIX=$SUFFIX-${OPT}_h
	if [[ $BETA1 != 0.9 ]]; then
		SUFFIX=${SUFFIX}_m$BETA1
	fi
	if [[ $ADAMBETA1 != 0.95 ]]; then
		SUFFIX=${SUFFIX}_b$ADAMBETA1
	fi
	if [[ ! -z "${MUON_LR_FACTOR+xxx}" ]]; then
		SUFFIX=${SUFFIX}_mlr$MUON_LR_FACTOR
		OPT_ARGS+=(--muon-lr-factor $MUON_LR_FACTOR)
	fi
	if [[ $OPT = dmuon ]]; then
		OPT=dist_muon
	fi
	if [[ $MUON_SCALE_MODE != spectral ]]; then
		if [[ $MUON_SCALE_MODE = unit_rms_norm ]]; then
			SUFFIX=${SUFFIX}_urm
		else
			SUFFIX=${SUFFIX}_$MUON_SCALE_MODE
		fi
	fi
	if [[ $MUON_NESTEROV = true ]]; then
		SUFFIX=${SUFFIX}_nest
		OPT_ARGS+=(--muon-use-nesterov)
	fi
	MUON_MOMENTUM=$BETA1
	BETA1=$ADAMBETA1
elif [[ $OPT = dmaster ]] || [[ $OPT = master ]]; then
	SUFFIX=$SUFFIX-${OPT}_h
	IS_MASTER_OPT=true
	if [[ $MASTER_ORTHOGONALIZE = true ]]; then
		SUFFIX=${SUFFIX}_o
		if [[ $BETA1 != 0.95 ]]; then
			SUFFIX=${SUFFIX}_m$BETA1
		fi
		if [[ $ADAMBETA1 != 0.95 ]]; then
			SUFFIX=${SUFFIX}_b$ADAMBETA1
		fi
		if [[ ! -z "${MUON_LR_FACTOR+xxx}" ]]; then
			SUFFIX=${SUFFIX}_mlr$MUON_LR_FACTOR
			OPT_ARGS+=(--muon-lr-factor $MUON_LR_FACTOR)
		fi
		OPT_ARGS+=(--use-orthogonal-updates)
		MUON_MOMENTUM=$BETA1
		BETA1=$ADAMBETA1
		if [[ $MUON_SCALE_MODE != spectral ]]; then
			if [[ $MUON_SCALE_MODE = unit_rms_norm ]]; then
				SUFFIX=${SUFFIX}_urm
			else
				SUFFIX=${SUFFIX}_$MUON_SCALE_MODE
			fi
		fi
	else
		if [[ $BETA1 != 0.9 ]] || [[ $BETA2 != 0.99 ]] || [[ $BETA3 != 0.999 ]]; then
			SUFFIX=${SUFFIX}_b${BETA1}_${BETA2}_$BETA3
		fi
	fi
	if [[ $ALPHA != 5 ]]; then
		SUFFIX=${SUFFIX}_a$ALPHA
	fi
	if [[ $OPT = dmaster ]]; then
		OPT=dist_master
	fi
elif [[ $OPT = ademamix ]]; then
	SUFFIX=$SUFFIX-amix
	OPT_ARGS+=(--overlap-grad-reduce)
	if [[ $BETA1 != 0.9 ]] || [[ $BETA2 != 0.95 ]] || [[ $BETA3 != 0.999 ]]; then
		SUFFIX=${SUFFIX}_b${BETA1}_${BETA2}_$BETA3
	fi
	if [[ $ALPHA != 5 ]]; then
		SUFFIX=${SUFFIX}_a$ALPHA
	fi
	if [[ $HYPERBALL = false ]]; then
		OPT_ARGS+=(--use-distributed-optimizer)
	fi
fi

if [[ $WEIGHT_DECAY != 0.1 ]]; then
	SUFFIX=$SUFFIX-wd$WEIGHT_DECAY
fi
if [[ $WEIGHT_DECAY_METHOD != decoupled ]]; then
	if [[ $IS_MASTER_OPT != true ]]; then
		echo "different weight decay method only implemented in the master opt"
		exit 1
	fi
	SUFFIX=$SUFFIX-$WEIGHT_DECAY_METHOD
fi

if [[ $HYPERBALL != false ]]; then
	if [[ $IS_MASTER_OPT != true ]]; then
		echo "hypersphere only implemented for master optimizer"
		exit 1
	fi
	SUFFIX=$SUFFIX-HS${HYPERBALL}${HS_R}_${HS_KIND}_it0
	OPT_ARGS+=(--hypersphere-mode $HYPERBALL --hypersphere-kind $HS_KIND --hypersphere-radius $HS_R)
	if [[ $HS_UPDATE = true ]]; then
		SUFFIX=${SUFFIX}_u
	else
		OPT_ARGS+=(--hypersphere-no-update)
	fi
	if [[ $HS_EMBED = true ]]; then
		SUFFIX=${SUFFIX}_emb
		OPT_ARGS+=(--hypersphere-embeddings)
		if [[ $HS_EMBED_NO_ORTHOGONAL = true ]]; then
			SUFFIX=${SUFFIX}NO
			OPT_ARGS+=(--no-use-orthogonal-embeddings)
		fi
	fi
	if [[ $HS_SPLIT_HEADS = true ]]; then
		SUFFIX=${SUFFIX}_sh
		OPT_ARGS+=(--hypersphere-split-heads)
	fi
	if [[ $HS_PROJECT = true ]]; then
		SUFFIX=${SUFFIX}_p
		OPT_ARGS+=(--hypersphere-project)
	fi
	if [[ $HS_SOFT = true ]]; then
		SUFFIX=${SUFFIX}_s
		OPT_ARGS+=(--hypersphere-soft)
	fi
fi

if [[ $CHANGED_LR = true ]]; then
	SUFFIX=$SUFFIX-lr$LR
fi

# FP8 settings.
FP8_ARGS=()
if [[ $FP8 = true ]]; then
	SUFFIX=$SUFFIX-fp8
	FP8_ARGS+=(--fp8-margin 0 --fp8-format hybrid --fp8-amax-history-len 1024 --fp8-amax-compute-algo max --fp8-recipe delayed)
fi

# Arch settings.
ARCH_ARGS=()
if [[ ! -z ${NEW_INIT_STD+x} ]]; then
	SUFFIX=$SUFFIX-std$NEW_INIT_STD
	INIT_STD=$NEW_INIT_STD
fi

if [[ $ACTIVATION = gelu ]]; then
	FFN_SIZE=$((3*$FFN_SIZE/2))
	SUFFIX=$SUFFIX-$ACTIVATION
elif [[ $ACTIVATION = swiglu ]]; then
	ARCH_ARGS+=(--swiglu)
else
	>&2 echo Unknown activation: $ACTIVATION
	exit 1
fi

if [[ $NORMALIZATION != RMSNorm ]]; then
	SUFFIX=$SUFFIX-$NORMALIZATION
fi
if [[ $NO_LEARNABLE_NORMS = true ]]; then
	SUFFIX=$SUFFIX-fz
	ARCH_ARGS+=(--no-learnable-norms)
fi
if [[ ! -z "${QK_NORM+xxx}" ]]; then
	if [[ $QK_NORM = RMSNorm ]]; then
		if [[ $NORMALIZATION != RMSNorm ]]; then
			echo When using qkRMSNorm you must also use --normalization RMSNorm
			exit 1
		fi
		SUFFIX=$SUFFIX-qkRMS
		ARCH_ARGS+=(--qk-layernorm)
	elif [[ $QK_NORM = L2Norm ]]; then
		SUFFIX=$SUFFIX-qkL2
		ARCH_ARGS+=(--qk-l2-norm)
	fi
fi

if [[ $NO_PRE_NORM = true ]]; then
	SUFFIX=$SUFFIX-nPre
	ARCH_ARGS+=(--no-pre-norm)
fi
if [[ $NO_FINAL_LAYERNORM = true ]]; then
	SUFFIX=$SUFFIX-nFin
	ARCH_ARGS+=(--no-final-layernorm)
fi
if [[ $POST_NORM = true ]]; then
	SUFFIX=$SUFFIX-pst
	ARCH_ARGS+=(--post-norm)
fi
if [[ $POST_BLOCK_NORM = true ]]; then
	SUFFIX=$SUFFIX-ppst
	ARCH_ARGS+=(--post-block-norm)
fi
if [[ $USE_STREAM_MINUS_RESIDUAL = true ]]; then
	SUFFIX=$SUFFIX-usmr
	ARCH_ARGS+=(--use-stream-minus-residual)
fi
LONG_SUFFIX=$SUFFIX  # To save checkpoints haha.
if [[ ! -z "${SOFT_MAX_SCALE+xxx}" ]]; then
	SUFFIX=$SUFFIX-ss
	LONG_SUFFIX=$LONG_SUFFIX-ss$SOFT_MAX_SCALE
	ARCH_ARGS+=(--softmax-scale $SOFT_MAX_SCALE)
fi

if [[ ! -z "${LAYER_SCALE+xxx}" ]]; then
	SUFFIX=$SUFFIX-ls
	LONG_SUFFIX=$LONG_SUFFIX-ls$LAYER_SCALE
	ARCH_ARGS+=(--layer-scale $LAYER_SCALE)
	if [[ ! -z "${LAYER_SCALE_SCALE+xxx}" ]]; then
		SUFFIX=${SUFFIX}S
		LONG_SUFFIX=${LONG_SUFFIX}S$LAYER_SCALE_SCALE
		ARCH_ARGS+=(--layer-scale-scale $LAYER_SCALE_SCALE)
	fi
fi

if [[ ! -z "${RESIDUAL_LAYER_SCALE+xxx}" ]]; then
	SUFFIX=$SUFFIX-rls
	LONG_SUFFIX=$LONG_SUFFIX-rls$RESIDUAL_LAYER_SCALE
	ARCH_ARGS+=(--residual-layer-scale $RESIDUAL_LAYER_SCALE)
	if [[ ! -z "${RESIDUAL_LAYER_SCALE_SCALE+xxx}" ]]; then
		SUFFIX=${SUFFIX}S
		LONG_SUFFIX=${LONG_SUFFIX}S$RESIDUAL_LAYER_SCALE_SCALE
		ARCH_ARGS+=(--residual-layer-scale-scale $RESIDUAL_LAYER_SCALE_SCALE)
	fi
fi

if [[ ! -z "${QK_LAYER_SCALE+xxx}" ]]; then
	SUFFIX=$SUFFIX-qkls
	LONG_SUFFIX=$LONG_SUFFIX-qkls$QK_LAYER_SCALE
	ARCH_ARGS+=(--qk-layer-scale $QK_LAYER_SCALE)
	if [[ ! -z "${QK_LAYER_SCALE_SCALE+xxx}" ]]; then
		LONG_SUFFIX=${LONG_SUFFIX}S$QK_LAYER_SCALE_SCALE
		SUFFIX=${SUFFIX}S
		ARCH_ARGS+=(--qk-layer-scale-scale $QK_LAYER_SCALE_SCALE)
	fi
fi
if [[ ! -z "${MLP_LAYER_SCALE+xxx}" ]]; then
	SUFFIX=$SUFFIX-mlpls
	LONG_SUFFIX=$LONG_SUFFIX-mlpls$MLP_LAYER_SCALE
	ARCH_ARGS+=(--mlp-layer-scale $MLP_LAYER_SCALE)
	if [[ ! -z "${MLP_LAYER_SCALE_GATE_SCALE+xxx}" ]]; then
		SUFFIX=${SUFFIX}G
		LONG_SUFFIX=${LONG_SUFFIX}G$MLP_LAYER_SCALE_GATE_SCALE
		ARCH_ARGS+=(--mlp-layer-scale-gate-scale $MLP_LAYER_SCALE_GATE_SCALE --no-bias-swiglu-fusion)
	fi
else
	if [[ ! -z "${MLP_LAYER_SCALE_GATE_SCALE+xxx}" ]]; then
		SUFFIX=${SUFFIX}-mlpG
		LONG_SUFFIX=${LONG_SUFFIX}-mlpG$MLP_LAYER_SCALE_GATE_SCALE
		ARCH_ARGS+=(--mlp-layer-scale-gate-scale $MLP_LAYER_SCALE_GATE_SCALE --no-bias-swiglu-fusion)
	fi
fi
if [[ ! -z "${MLP_OUT_SCALE+xxx}" ]]; then
	SUFFIX=$SUFFIX-mlpO
	LONG_SUFFIX=$LONG_SUFFIX-mlpO$MLP_OUT_SCALE
	ARCH_ARGS+=(--mlp-out-scale $MLP_OUT_SCALE)
fi
if [[ ! -z "${LOGITS_LAYER_SCALE+xxx}" ]]; then
	SUFFIX=$SUFFIX-lgsls
	LONG_SUFFIX=$LONG_SUFFIX-lgsls$LOGITS_LAYER_SCALE
	ARCH_ARGS+=(--logits-layer-scale $LOGITS_LAYER_SCALE)
	if [[ ! -z "${LOGITS_LAYER_SCALE_SCALE+xxx}" ]]; then
		SUFFIX=${SUFFIX}S
		LONG_SUFFIX=${LONG_SUFFIX}S$LOGITS_LAYER_SCALE_SCALE
		ARCH_ARGS+=(--logits-layer-scale-scale $LOGITS_LAYER_SCALE_SCALE)
	fi
fi
if [[ ! -z "${UPSCALE_EMBEDDING+xxx}" ]]; then
	SUFFIX=$SUFFIX-ue
	LONG_SUFFIX=$LONG_SUFFIX-up$UPSCALE_EMBEDDING
	ARCH_ARGS+=(--upscale-embedding $UPSCALE_EMBEDDING)
fi


# Training settings.
if [[ $NO_WARMUP = true ]]; then
	SUFFIX=$SUFFIX-nw
	LONG_SUFFIX=$LONG_SUFFIX-nw
	WARMUP=0
else
	WARMUP=5000
fi
if [[ $TOKENS != $DEF_TOKENS ]]; then
	SUFFIX=$SUFFIX-${TOKENS}BT
	LONG_SUFFIX=$LONG_SUFFIX-${TOKENS}BT
fi
ITERS=$((ITERS_PER_BT*TOKENS))

DECAY_ARGS=()
if [[ $DECAY = wsd ]]; then
	if [[ $COOLDOWN != 0.2 ]]; then
		SUFFIX=$SUFFIX-cd$COOLDOWN
		LONG_SUFFIX=$LONG_SUFFIX-cd${COOLDOWN}
	fi
	DECAY_ITERS=$(python3 -c "print(int($ITERS * $COOLDOWN))")
	DECAY_ARGS+=(
		--lr-decay-style WSD
		--lr-wsd-decay-style minus_sqrt
		--lr-wsd-decay-iters $DECAY_ITERS
	)
elif [[ $DECAY = cos ]]; then
	SUFFIX=$SUFFIX-cos
	LONG_SUFFIX=$LONG_SUFFIX-cos
	DECAY_ITERS=$((ITERS - WARMUP))
	DECAY_ARGS+=(
		--lr-decay-style cosine
		--lr-decay-iters $DECAY_ITERS
	)
else
	echo "Unknown decay method $DECAY"
	exit 1
fi

# In order to make the name shorter, we will alias all the suffixes that correspond to
# ngpt architecture to just show ngpt.
NGPT_SUBSTRING=-L2Norm-fz-nPre-nFin-pst-ppst-usmr-ss-lsS-qklsS-mlplsG-lgslsS
SUFFIX="${SUFFIX/$NGPT_SUBSTRING/-ngpt}"

SUFFIX=$SUFFIX$EXTRA_NAME
LONG_SUFFIX=$LONG_SUFFIX$EXTRA_NAME
EXTRA_LOGS=()
if [[ $EXTRA_LOG = true ]]; then
	EXTRA_LOGS+=(
		--log-validation-ppl-to-tensorboard
		--log-params-norm-per-param
		--log-num-zeros-in-grad
		--log-params-norm
		--log-progress
		--log-timers-to-tensorboard
		--log-model-internals
		--log-activation-stats
		--log-gradient-stats
		--log-angular-metrics
		--log-relative-updates
		--log-delta-y
		--internals-log-interval $LOG_FREQ
	)
fi

# Final preparations.
WANDB_PROJECT=opt_$SCRIPT_VERSION
EXP_NAME=$SIZE$SCALE$SUFFIX
ROOT_PATH=$TRAIN_ROOT/$SCRIPT_VERSION/$SIZE$SCALE$LONG_SUFFIX
DEBUG_ROOT=$ROOT_PATH/debug
SAVE_PATH=$ROOT_PATH/checkpoints
DIFFS_PATH=$ROOT_PATH/diffs
TRIGGER_DIR=$ROOT_PATH/triggerdir
TRIGGER_LOCK=$TRIGGER_DIR/training_lock
mkdir -p $SAVE_PATH
mkdir -p $DIFFS_PATH
mkdir -p $TRIGGER_DIR

if [[ -z ${WANDB_NAME+x} ]]; then
	WANDB_NAME=$EXP_NAME
fi

if [[ ${#WANDB_NAME} -gt 117 ]]; then
	# We compare against 117 because WANDB_NAME gets appended -n$NODE_COUNT-$JOB_ID in the final sbatch.
	>&2 echo "WANDB_NAME is too long (it shouldn't exceed 117 characters): $WANDB_NAME"
	exit 1
fi

#= WRAPPING UP: Set up the _ARGS variables that are going to be used in the end =#

LLAMA_ARGS=(
	--tensor-model-parallel-size $TP
	--pipeline-model-parallel-size $PP
	$MAYBE_VIRTUAL
	--seq-length $SEQ_LEN
	--max-position-embeddings $SEQ_LEN
	--tokenizer-type HuggingFaceTokenizer
	--tokenizer-model $TOKENIZER
	--normalization $NORMALIZATION
	--position-embedding-type rope
	--attention-softmax-in-fp32
	--disable-bias-linear
	--transformer-impl transformer_engine
	--num-layers $LAYERS
	--hidden-size $HIDDEN_SIZE
	--group-query-attention
	--num-query-groups $NUM_QUERY_GROUPS
	--ffn-hidden-size $FFN_SIZE
	--num-attention-heads $NUM_HEADS
	--attention-dropout 0.0
	--hidden-dropout 0.0
	--rotary-base 500000
	--rotary-percent 1.0
	--use-rope-scaling
	--bf16
	--optimizer $OPT
	--adam-eps 0.00000001
	--norm-epsilon 0.00001
)
if [[ $UNTIE = true ]]; then
	LLAMA_ARGS+=(--untie-embeddings-and-output-weights)
fi

TRAINING_ARGS=(
	--micro-batch-size $MBS
	--global-batch-size $GBS
	--train-iters $ITERS
	--eval-interval $((ITERS + 1))
	--eval-iters 1
	--weight-decay $WEIGHT_DECAY
	--weight-decay-method $WEIGHT_DECAY_METHOD
	--adam-beta1 $BETA1
	--muon-momentum $MUON_MOMENTUM
	--muon-scale-mode $MUON_SCALE_MODE
	--adam-beta2 $BETA2
	--ademamix-beta3 $BETA3
	--ademamix-alpha $ALPHA
	--ademamix-beta3-warmup $ITERS
	--ademamix-alpha-warmup $ITERS
	--init-method-std $INIT_STD
	--clip-grad 1.0
	--lr $LR
	--min-lr $MIN_LR
	--trigger-path $TRIGGER_DIR
	--exit-signal-handler
)

DATA_ARGS=(
	--data-path $DATA_PATHS
	--data-cache-path $SCRATCH/data/cache_opt
	--split 100,0,0
)
	#--mock-data

LOGGING=(
	--log-interval 1
	--load $SAVE_PATH
	--tensorboard-dir $ROOT_PATH/tensorboard
	--wandb-project $WANDB_PROJECT
	--wandb-save-dir $ROOT_PATH/wandb
	--timing-log-level 1
	--tensorboard-log-interval 1
	--log-throughput
)
if [[ $NO_SAVE = false ]]; then
	LOGGING+=(
		--save-interval $SAVE_FREQ
		--save $SAVE_PATH
	)
fi
LOGGING=(${LOGGING[@]} ${EXTRA_LOGS[@]})

SCHEDULER_ARGS=(
	--lr-warmup-iters $WARMUP
)

EXTRA_ARGS+=(
	--async-save
)

ARGS="${LLAMA_ARGS[@]} ${TRAINING_ARGS[@]} ${SCHEDULER_ARGS[@]} ${DATA_ARGS[@]} ${LOGGING[@]} ${EXTRA_ARGS[@]} ${FP8_ARGS[@]} ${ARCH_ARGS[@]} ${OPT_ARGS[@]} ${DECAY_ARGS[@]}"

#= RUNNING: Prepare and launch a slurm script =#
CMD="numactl --membind=0-3 env python3 pretrain_gpt.py $ARGS"

mkdir -p $ROOT_PATH
cat > $ROOT_PATH/submission.sbatch <<- EOM
#!/bin/bash
#SBATCH --time=$TIME
#SBATCH --job-name=$EXP_NAME
#SBATCH --output=$ROOT_PATH/slurmlogs/%j.out
#SBATCH --error=$ROOT_PATH/slurmlogs/%j.err
#SBATCH --nodes=$NODES
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=36
#SBATCH --mem=460000
#SBATCH --exclusive
#SBATCH --account=a-infra01-1
#SBATCH --partition=${PARTITION:-normal}
#SBATCH --signal=SIGTERM@180
#SBATCH --dependency=singleton
#SBATCH --environment=$CONTAINER 

# Wake up.
echo [\$(date)] Starting job
echo [\$(date)] Using nodes: \$SLURM_JOB_NODELIST
srun -l bash -c 'echo \$(hostname) \$(nvidia-smi | grep -o "|\\s*[0-9]*MiB")'


# Log git status.
cd $CODE_PATH
echo ---------
echo git status:
git status
echo git log:
git log -n 1
echo ---------
git diff > $DIFFS_PATH/\$SLURM_JOBID.diff

export MASTER_ADDR=\$(hostname)
export WORLD_SIZE=\$SLURM_NPROCS
export MASTER_PORT=25678

export WANDB__FILE_STREAM_RETRY_MAX=10
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK

export TRITON_HOME=$TRITON_HOME_DIR
export TRITON_CACHE_DIR=$TRITON_HOME_DIR/cache

# debug!
DEBUG_DIR=$DEBUG_ROOT/\$SLURM_JOB_ID
mkdir -p \$DEBUG_DIR
cp \$0 \$DEBUG_DIR/submit.sbatch
cat \$SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment > \$DEBUG_DIR/env.toml
echo "\\nMegatron path: $CODE_PATH (\$(git -C $CODE_PATH rev-parse --verify HEAD))" > \$DEBUG_DIR/git
git diff > \$DEBUG_DIR/git.diff
pip list > \$DEBUG_DIR/pip.txt
nvidia-smi > \$DEBUG_DIR/cuda
printenv > \$DEBUG_DIR/env.sh

srun -lu --cpus-per-task \$SLURM_CPUS_PER_TASK bash -c "
	cd $CODE_PATH
	export PYTHONPATH=\$PWD:$EMERGING_OPTIMIZERS_PATH
	export RANK=\\\$SLURM_PROCID
	export LOCAL_RANK=\\\$SLURM_LOCALID
	mkdir -p \\\$TRITON_CACHE_DIR
	$CMD --wandb-exp-name $EXP_NAME-n$NODES-\\\$SLURM_JOBID
"

# Goodbye lol.
echo [\$(date)] Goodbye
EOM
echo "Saved sbatch to $ROOT_PATH/submission.sbatch"

sbatch $ROOT_PATH/submission.sbatch
