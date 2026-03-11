source .env  # export WANDB_API_KEY and HF_TOKEN here.

# Best llama arch muon run we have (5dd6jz0k).
# All previous ablations get messy and are not directly comparable cuz we e.g. had the no splitting of the heads bug, spectral scale, etc.
# Regardless, with the no splitting of the heads bug, we had already ablated:
# - 1x muon lr (8iq06ryh&n5lt0kst)
# - 1x muon lr & nesterov (sz9fpncd&g3vrnm4l)
# - 2x muon lr (8oiw20ul&08gqwurv)
# - 2x muon lr & nesterov (179kpk2c&8jikbr4q)
# - 4x muon lr (qph9iahj&r0w1rktt)
# - 1x muon lr & 0.01 base lr (cxqmww0d&a24glhdw)
# - 1x muon lr & base lr & nesterov (umaavm3t&q91pm0ch)
# The trend from those runs generally: no nesterov > nesterov, 2x muon lr is best, 0.01 base lr is bad (but stable, surprisingly).
# Potential sweeps we want to do with the final config: base lr, muon momentum & adam beta1 tho we have (some) of those ablations with the spectral scale and 1x muon lr, maybe they transfer lol.
# 
# All of the following ablations follow spectral scale and 1x muon lr (and also no split head bug):
# - 0.5 base lr (wx3wj1s3&2ozoyywy)
# - 1x base lr (przqq3kgesf8g3q5)
# - 2x base lr (vvr0fm72&ufcsehdi)
# - 4x base lr (c70jlcjr) - diverged
# - 6x base lr (2m0tjy97) - diverged
# - 4x base lr & 0.9 beta1 (d7g8exm9&j1fyqsdb)
# In terms of lr ablations: 4x was best.
# In terms of beta1: 0.95 (default) was best.
bash submissions/submit.sh 110 --opt muon --b1 0.95 --mlr 2 --muon-scale unit_rms_norm $*
