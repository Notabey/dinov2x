#!/bin/bash

# 设置 PYTHONPATH 环境变量
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0,1

# 启动训练脚本
torchrun \
    --nproc_per_node=2 \
    dinov2/run/train/train.py \
    --no-submitit \
    --nodes 1 \
    --ngpus=2 \
    --config-file dinov2/configs/train/vitl16_short.yaml \
    --output-dir output/ \
    --no-resume \
    "train.dataset_path=ImageNet;split=TRAIN;root=/root/autodl-tmp/imagenet;extra=/root/autodl-tmp/imagenet;webdataset=no"