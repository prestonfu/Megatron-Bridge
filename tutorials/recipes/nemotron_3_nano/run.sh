trap 'exit 1' INT

run() {
  local name="$1"; shift
  RUN_ID=$(date +%y%m%d_%H%M%S)
  torchrun --nproc_per_node=8 tutorials/recipes/nemotron_3_nano/02_pretrain_with_yaml.py \
    --config-file tutorials/recipes/nemotron_3_nano/conf/nemotron_3_nano_small_pretrain_fineweb_muon.yaml \
    checkpoint.save=/mnt/checkpoints/megatron_bridge/nemotron_3_nano_small_muon/$RUN_ID \
    train.train_iters=1010 \
    scheduler.lr_decay_iters=1010 \
    scheduler.lr_wsd_decay_iters=0 \
    logger.wandb_exp_name="$name" \
    "$@"
}

run_adamw() {
  local name="$1"; shift
  RUN_ID=$(date +%y%m%d_%H%M%S)
  torchrun --nproc_per_node=8 tutorials/recipes/nemotron_3_nano/02_pretrain_with_yaml.py \
    --config-file tutorials/recipes/nemotron_3_nano/conf/nemotron_3_nano_small_pretrain_fineweb_adamw.yaml \
    checkpoint.save=/mnt/checkpoints/megatron_bridge/nemotron_3_nano_small_adamw/$RUN_ID \
    train.train_iters=1010 \
    scheduler.lr_decay_iters=1010 \
    scheduler.lr_wsd_decay_iters=0 \
    logger.wandb_exp_name="$name" \
    "$@"
}

run small_muon_lr5e-4 \
  optimizer.lr=0.0005 \
  train.train_iters=10010 \
  scheduler.lr_decay_iters=10010 \
  scheduler.lr_wsd_decay_iters=2010 \
  checkpoint.save_interval=1000 || true

# run small_muon_lr3e-4_nosoftcap \
#   optimizer.lr=0.0003 \
#   '~model.attn_logit_softcap' \
#   '~model.output_logit_softcap' || true

# run small_muon_lr5e-4 \
#   optimizer.lr=0.0005 || true

# run small_muon_lr1e-3 \
#   optimizer.lr=0.001 || true

# run small_muon_polar_lr3e-4 \
#   optimizer.lr=0.0003 \
#   optimizer.muon_coefficient_type=polar_express \
#   optimizer.muon_num_ns_steps=8 || true

# run_adamw small_adamw_lr3e-4 \
#   optimizer.lr=0.0003 || true

# run_adamw small_adamw_lr5e-4 \
#   optimizer.lr=0.0005 || true

# run_adamw small_adamw_lr1e-3 \
#   optimizer.lr=0.001 || true
