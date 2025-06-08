@echo off
set PYTHONPATH=.

python dinov2\run\train\train.py ^
    --no-submitit ^
    --nodes 1 ^
    --config-file dinov2\configs\train\vitl16_short.yaml ^
    --output-dir output\ ^
    "train.dataset_path=ImageNet;split=TRAIN;root=D:\Pictures\DataSets\imagenet-1k-wds;extra=D:\Pictures\DataSets\imagenet-1k-wds;webdataset=yes"

exit /b 0