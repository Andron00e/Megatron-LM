source .env  # export WANDB_API_KEY and HF_TOKEN here.
INV_LAYERS=0.083  # approx 1/n_layers
INV_SQRTMODELDIM=0.044  # approx 1/sqrt(hidden_dim)
SQRT_KDIM=11.31  # approx sqrt(k_dim)
INV_SQRTKDIM=0.088  # approx 1/sqrt(k_dim)
SQRT_MODELDIM=22.62  # approx sqrt(hidden_dim)

# Best ngpt baseline we have (tduik3hz&l6mu9dkt).
# Since there are many components, previous ablations don't necessarily match this exact config.
# Whenever possible, I provide alehc/opt_v1 wandb IDs.
# Importantly, all previous results were done without qkv_splitting and no normalization at initialization, which actually had slightly better loss) (4jqht7p1&mc6mvxpg).
# Other than that, we already ablated:
#  - 2x LR: had almost same exact loss (yf4u8w2k&brikbri4).
#  - WSD decay: much worse loss (86npb305&ishzhid8).
# With WSD decay (and all other things the same), we already ablated:
#  - 0.5x LR & 2x LR.
bash submissions/submit.sh 110 \
	--no-pre-norm --no-final-layernorm \
	--opt master --alpha 0 --hs row --hs-embed --hs-split-heads \
	--normalization L2Norm --no-learnable-norms --post-norm --post-block-norm --use-stream-minus-residual --layer-scale $INV_LAYERS --layer-scale-scale $INV_SQRTMODELDIM \
	--softmax-scale $SQRT_KDIM --qk-layer-scale 1 --qk-layer-scale-scale $INV_SQRTKDIM \
	--mlp-layer-scale 1 --mlp-layer-scale-gate-scale $SQRT_MODELDIM \
	--logits-layer-scale 1 --logits-layer-scale-scale $INV_SQRTMODELDIM \
	--no-warmup --wd 0 \
	--decay cos \
	--init $INV_SQRTMODELDIM \
	$*
