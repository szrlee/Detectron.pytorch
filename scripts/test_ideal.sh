#!/bin/bash

dir=Outputs/e2e_faster_rcnn_R-101-FPN_VOC-clipart-ideal/Oct11-06-41-35_ip-172-31-8-158_step/ckpt
config=configs/dt/e2e_faster_rcnn_R-101-FPN_VOC-clipart-ideal.yaml
dataset=clipart
for iter in $(eval echo "{499..5999..500}")
do
    echo "Start testing on iteration $iter"
    CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test_net.py --dataset $dataset --cfg $config --load_ckpt $dir/model_step$iter.pth --multi-gpu-testing
done

