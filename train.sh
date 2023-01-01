#!/bin/sh
PARTITION=Segmentation

GPU_ID=0
dataset=coco # pascal coco
exp_name=split0

arch=BAM
net=resnet50 # vgg resnet50

exp_dir=/app/BAM/exp/${dataset}/${arch}/${exp_name}/${net} #pathを変更してます。
snapshot_dir=${exp_dir}/snapshot
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}_${net}.yaml
mkdir -p ${snapshot_dir} ${result_dir}
now=$(date +"%Y%m%d_%H%M%S")
cp train.sh train.py ${config} ${exp_dir}

echo ${arch}
echo ${config}

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 train.py \
        --config=${config} \
        --arch=${arch} \
        2>&1 | tee ${result_dir}/train-$now.log