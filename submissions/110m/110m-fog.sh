source .env  # export WANDB_API_KEY and HF_TOKEN here.
# 1/sqrt(L)
INV_SQRT_L=0.28  # slightly less to avoid blow ups at initialization.

# Best FOG-swiglu architecture & adamw run (upmrn8e8).
bash submissions/submit.sh 110 \
	--no-pre-norm --post-norm --no-final-layernorm \
	--layer-scale $INV_SQRT_L \
	--qk-norm RMSNorm \
	$*
