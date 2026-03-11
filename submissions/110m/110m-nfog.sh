source .env  # export WANDB_API_KEY and HF_TOKEN here.
# 1/sqrt(L)
INV_SQRT_L=0.28  # slightly less to avoid blow ups at initialization.

# Default FOG-swiglu hyper adamw settings.
bash submissions/submit.sh 110 \
	--opt master --alpha 0 --hs row --hs-embed --hs-split-heads \
	--no-pre-norm --post-norm --no-final-layernorm \
	--layer-scale $INV_SQRT_L \
	--qk-norm RMSNorm \
	--logits-layer-scale 1 \
	--wd 0 --no-warmup --decay cos \
	$*
