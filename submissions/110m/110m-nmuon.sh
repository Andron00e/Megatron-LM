source .env  # export WANDB_API_KEY and HF_TOKEN here.

# Best ngpt arch hypermuon config (45ynu08h&308ebhvk).
# Note that this config isn't reproducible with the current scripts anymore, as it was done before the qkv_split & normalization init fixes.
# After the fixes the same thing (3yhae95i&6aia7ss3) runs much worse for some reason, TODO investigate I guess.
# We have ablated a bunch of things here but we can't seem to improve upon this.
# Some attempts include:
# - unit_rms_norm scaling & mlr2 & no update normalization (c57fa6o4&v8jpqdp7), which was just slightly worse.
# - normalization of embeddings but using adam on them (ee2898mg&8gtxa1c8), not to far behind.
# - no update normalization and no scaling either, somewhat worse.
# - a few options without update norm & but with embeddings norm using adam on them, all quite bad.
# - 0.5x & 1x lr sweep.
# - 0.95 beta1.
# - sinkhorn which was very good in the beginning but quite ugly in the cooldown.

INV_LAYERS=0.083  # approx 1/n_layers
INV_SQRTMODELDIM=0.044  # approx 1/sqrt(hidden_dim)
SQRT_KDIM=11.31  # approx sqrt(k_dim)
INV_SQRTKDIM=0.088  # approx 1/sqrt(k_dim)
SQRT_MODELDIM=22.62  # approx sqrt(hidden_dim)
bash submissions/submit.sh 110 \
	--no-pre-norm --no-final-layernorm \
	--opt master --master-orthogonalize --hs row --hs-split-heads --hs-u \
	--b1 0.95 --mb1 0.9 --lr 0.004 \
	--normalization L2Norm --no-learnable-norms --post-norm --post-block-norm --use-stream-minus-residual --layer-scale $INV_LAYERS --layer-scale-scale $INV_SQRTMODELDIM \
	--softmax-scale $SQRT_KDIM --qk-layer-scale 1 --qk-layer-scale-scale $INV_SQRTKDIM \
	--mlp-layer-scale 1 --mlp-layer-scale-gate-scale $SQRT_MODELDIM \
	--logits-layer-scale 1 --logits-layer-scale-scale $INV_SQRTMODELDIM \
	--no-warmup --wd 0 \
	--init $INV_SQRTMODELDIM \
	$*
