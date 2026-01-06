export NCCL_IB_DISABLE=0
export NCCL_IB_TIMEOUT=3600 
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600 
export NCCL_LAUNCH_MODE=PARALLEL 
export NCCL_IB_HCA=mlx5 
export NCCL_IB_TC=136 
export NCCL_IB_SL=5 
export NCCL_IB_GID_INDEX=3 
export NCCL_SOCKET_IFNAME=bond1
# export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export SETUPTOOLS_USE_DISTUTILS=stdlib
# export WANDB_RESUME=auto
# export WANDB_RUN_ID=uzh3m4s6

# multi-node training in lightning style, e.g., 4 nodes
export MASTER_ADDR=29.111.44.202
export MASTER_PORT=28778
export NNODES=4
export NGPUS_PER_NODE=8
export NODE_RANK=${NODE_RANK:-0}

# Run experiments from coarse to fine (reversed order)
# Phase 1: interval of 4 (22, 18, 14, 10, 6)
# Phase 2: remaining layers (24, 20, 16, 12, 8)
echo "=== Starting experiments from layer 24 to layer 6 ==="
for layer in 24 20 16 8 22 18 14 10 6 12; do
    echo "Running experiment for layer ${layer}..."
    python main.py fit -c configs_flow/internvit_2b_layer${layer}.yaml \
        --trainer.num_nodes=4 \
        --trainer.devices=8 \
        --trainer.strategy=ddp
    echo "Completed layer ${layer}"
    echo "---"
    sleep 30
done

echo "=== All experiments completed! ==="