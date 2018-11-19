#!/bin/bash
dir=Outputs/e2e_faster_rcnn_R-101-FPN_clipart_weakly_1stream_noreg_freeze-boxhead_unlock-rpn-conv/Nov13-17-06-52_ip-172-31-8-158_step/ckpt
config=configs/dt/e2e_faster_rcnn_R-101-FPN_clipart_weakly_1stream_noreg_freeze-boxhead_unlock-rpn-conv.yaml
dataset=clipart
iter=4999
echo "Start testing on iteration $iter"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/test_net.py --dataset $dataset --vis --cfg $config --load_ckpt $dir/model_step$iter.pth --multi-gpu-testing
