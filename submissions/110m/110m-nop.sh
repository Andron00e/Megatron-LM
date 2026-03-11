source .env  # export WANDB_API_KEY and HF_TOKEN here.
# 1/sqrt(L)
INV_SQRT_L=0.28  # slightly less to avoid blow ups at initialization.

# Default OP with hypersphere adam.
# Blows up with SwiGLU activation.
bash submissions/submit.sh 110 \
	--opt master --alpha 0 --hs row --hs-embed --hs-split-heads \
	--no-pre-norm --no-final-layernorm \
	--layer-scale $INV_SQRT_L \
	--qk-norm RMSNorm \
	--activation gelu \
	--logits-layer-scale 1 \
	--wd 0 --no-warmup --decay cos \
	$*
