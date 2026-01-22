export NCCL_IB_DISABLE=0
# export NCCL_IB_TIMEOUT=3600 
# export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600 
# export NCCL_LAUNCH_MODE=PARALLEL 
# export NCCL_IB_HCA=mlx5 
# export NCCL_IB_TC=136 
# export NCCL_IB_SL=5 
# export NCCL_IB_GID_INDEX=3 
export NCCL_SOCKET_IFNAME=bond1
# export NCCL_DEBUG=INFO
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export SETUPTOOLS_USE_DISTUTILS=stdlib
# export WANDB_RESUME=auto
# export WANDB_RUN_ID=f7wp6p2f
export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"
# multi-node training in lightning style, e.g., 4 nodes
export MASTER_ADDR=29.111.44.202
export MASTER_PORT=28778
export NNODES=4
export NGPUS_PER_NODE=8
export NODE_RANK=${NODE_RANK:-0}

export WANDB_RESUME=auto
export WANDB_RUN_ID=0u8tcxuc

python main.py fit -c  configs_flow/uniflow_internvit_2b_blip3o_rec_stage2.yaml \
    --trainer.num_nodes=4 \
    --trainer.devices=8 \
    --ckpt_path=dual_internvit_2b/exp_sem_gen_gate_c256_new_stage2/epoch=0-step=25000.ckpt