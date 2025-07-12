#!/bin/bash

export NCCL_IB_GID_INDEX=3
export NCCL_IB_DISABLE=0

export NCCL_IB_HCA=mlx5_0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=NET
export NCCL_SOCKET_IFNAME=ens8np0
export NCCL_NET_GDR_READ=1
export NCCL_NET_GDR_LEVEL=3


torchrun --nproc_per_node=6 --nnodes=3 --node_rank=0 --master_addr="" --master_port=12345 -m src.main

