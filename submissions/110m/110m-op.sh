source .env  # export WANDB_API_KEY and HF_TOKEN here.
# 1/sqrt(L)
INV_SQRT_L=0.28  # slightly less to avoid blow ups at initialization.

# Default OP run.
# Blows up with SwiGLU activation.
bash submissions/submit.sh 110 \
	--no-pre-norm --no-final-layernorm \
	--layer-scale $INV_SQRT_L \
	--qk-norm RMSNorm \
	--activation gelu \
	$*
