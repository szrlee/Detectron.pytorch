#!/bin/bash
dir=Outputs/e2e_faster_rcnn_R-101-FPN_clipart_weakly_2stream_noreg_copy_freeze-boxhead/Nov12-02-16-14_ip-172-31-8-158_step/ckpt
config=configs/dt/e2e_faster_rcnn_R-101-FPN_clipart_weakly_2stream_noreg_copy_freeze-boxhead.yaml
dataset=clipart

for iter in $(eval echo "{499..9999..500}")
do
    echo "Start testing on iteration $iter"
    CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test_net.py --dataset $dataset --cfg $config --load_ckpt $dir/model_step$iter.pth --multi-gpu-testing
done
