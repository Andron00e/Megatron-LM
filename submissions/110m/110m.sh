source .env  # export WANDB_API_KEY and HF_TOKEN here.

# Best Llama arch, adamw baseline we have found.
# Already ablated 0.5x and 2x LR.
bash submissions/submit.sh 110 $*
