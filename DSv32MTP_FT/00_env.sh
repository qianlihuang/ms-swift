#!/usr/bin/env bash
set -euo pipefail

# Basic runtime env
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Optional: use a dedicated log dir if needed
export SWIFT_LOG_LEVEL=info

# Optional: set OMP threads
export OMP_NUM_THREADS=8

# NOTE:
# If you run multi-node, also set:
#   export MASTER_ADDR=<ip>
#   export MASTER_PORT=<port>
#   export NODE_RANK=<rank>
#   export NNODES=<num_nodes>
#   export NPROC_PER_NODE=8
