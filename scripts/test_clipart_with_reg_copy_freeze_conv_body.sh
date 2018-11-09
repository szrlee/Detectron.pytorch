#!/usr/bin/env bash
dir=Outputs/e2e_faster_rcnn_R-101-FPN_clipart_weakly_with-spatial-reg_copy-cls-to-det_freeze-conv-body/Nov01-04-08-39_ip-172-31-8-158_step/ckpt/
config=configs/dt/e2e_faster_rcnn_R-101-FPN_clipart_weakly_with-spatial-reg_copy-cls-to-det_freeze-conv-body.yaml
dataset=clipart

for iter in $(eval echo "{499..9999..500}")
do
    echo "Start testing on iteration $iter"
    CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test_net.py --dataset $dataset --cfg $config --load_ckpt $dir/model_step$iter.pth --multi-gpu-testing
done

