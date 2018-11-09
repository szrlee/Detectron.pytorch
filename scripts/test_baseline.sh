#!/bin/bash

dir=Outputs/e2e_faster_rcnn_R-101-FPN_1x-VOC/Oct08-14-54-41_ip-172-31-8-158_step/ckpt
config=configs/baselines/e2e_faster_rcnn_R-101-FPN_1x-VOC.yaml
dataset=clipart
for iter in $(eval echo "{4999..79999..5000}")
do
    echo "Start testing on iteration $iter"
    CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test_net.py --dataset $dataset --cfg $config --load_ckpt $dir/model_step$iter.pth --multi-gpu-testing
done

