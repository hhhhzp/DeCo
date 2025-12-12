# multi-node training in lightning style, e.g., 4 nodes
export MASTER_ADDR=29.111.44.218
export MASTER_PORT=28778
export NNODES=4
export NGPUS_PER_NODE=8
NODE_RANK=${NODE_RANK:-0}
python main.py fit -c ./configs_c2i/DeCo_XL.yaml --trainer.num_nodes=4