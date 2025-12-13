export NCCL_IB_DISABLE=1
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

# multi-node training in lightning style, e.g., 4 nodes
export MASTER_ADDR=29.111.44.218
export MASTER_PORT=28778
export NNODES=4
export NGPUS_PER_NODE=8
export NODE_RANK=${NODE_RANK:-0}
python main.py fit -c configs_c2i/ReCo_large.yaml \
    --trainer.num_nodes=4 \
    --trainer.devices=8 \
    --trainer.strategy=ddp \
    --ckpt_path=./universal_pix_workdirs/exp_DeCo_256_large/epoch=7-step=20000.ckpt