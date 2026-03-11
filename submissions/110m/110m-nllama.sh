source .env  # export WANDB_API_KEY and HF_TOKEN here.
INV_LAYERS=0.083  # approx 1/n_layers
INV_SQRTMODELDIM=0.044  # approx 1/sqrt(hidden_dim)
SQRT_KDIM=11.31  # approx sqrt(k_dim)
INV_SQRTKDIM=0.088  # approx 1/sqrt(k_dim)
SQRT_MODELDIM=22.62  # approx sqrt(hidden_dim)

# Default nGPT architecture but adamw optimizer (no hypersphere constraint on weights).
bash submissions/submit.sh 110 \
	--no-pre-norm --no-final-layernorm \
	--normalization L2Norm --no-learnable-norms --post-norm --post-block-norm --use-stream-minus-residual --layer-scale $INV_LAYERS --layer-scale-scale $INV_SQRTMODELDIM \
	--softmax-scale $SQRT_KDIM --qk-layer-scale 1 --qk-layer-scale-scale $INV_SQRTKDIM \
	--mlp-layer-scale 1 --mlp-layer-scale-gate-scale $SQRT_MODELDIM \
	--logits-layer-scale 1 --logits-layer-scale-scale $INV_SQRTMODELDIM \
	--init $INV_SQRTMODELDIM \
	$*
