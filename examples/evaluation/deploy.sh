# Unset SLURM/PMI/PMIX env vars to prevent MPI initialization issues
for i in $(env | grep ^SLURM_ | cut -d"=" -f 1); do unset -v $i; done
for i in $(env | grep ^PMI_ | cut -d"=" -f 1); do unset -v $i; done
for i in $(env | grep ^PMIX_ | cut -d"=" -f 1); do unset -v $i; done

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
MEGATRON_CHECKPOINT=$1
python \
  /docker_workspace/hw2/Export-Deploy/scripts/deploy/nlp/deploy_ray_inframework.py \
  --megatron_checkpoint "$MEGATRON_CHECKPOINT" \
  --model_id megatron_model \
  --host 0.0.0.0 \
  --port 8000 \
  --num_gpus 8 \
  --num_replicas 1 \
  --tensor_model_parallel_size 1 \
  --pipeline_model_parallel_size 1 \
  --context_parallel_size 1 \
  --expert_model_parallel_size 8 \
  --legacy_ckpt
