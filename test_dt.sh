#!/usr/bin/env bash
dir=Outputs/e2e_faster_rcnn_R-101-FPN_dt-clipart-VOC/Nov01-08-07-07_ip-172-31-8-158_step/ckpt

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test_net.py --dataset clipart --cfg configs/dt/e2e_faster_rcnn_R-101-FPN_dt-clipart-VOC.yaml --load_ckpt $dir/model_step4999.pth --multi-gpu-testing
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test_net.py --dataset clipart --cfg configs/dt/e2e_faster_rcnn_R-101-FPN_dt-clipart-VOC.yaml --load_ckpt $dir/model_step9999.pth --multi-gpu-testing
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test_net.py --dataset clipart --cfg configs/dt/e2e_faster_rcnn_R-101-FPN_dt-clipart-VOC.yaml --load_ckpt $dir/model_step19999.pth --multi-gpu-testing
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test_net.py --dataset clipart --cfg configs/dt/e2e_faster_rcnn_R-101-FPN_dt-clipart-VOC.yaml --load_ckpt $dir/model_step29999.pth --multi-gpu-testing
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test_net.py --dataset clipart --cfg configs/dt/e2e_faster_rcnn_R-101-FPN_dt-clipart-VOC.yaml --load_ckpt $dir/model_step39999.pth --multi-gpu-testing
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test_net.py --dataset clipart --cfg configs/dt/e2e_faster_rcnn_R-101-FPN_dt-clipart-VOC.yaml --load_ckpt $dir/model_step49999.pth --multi-gpu-testing
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test_net.py --dataset clipart --cfg configs/dt/e2e_faster_rcnn_R-101-FPN_dt-clipart-VOC.yaml --load_ckpt $dir/model_step59999.pth --multi-gpu-testing
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test_net.py --dataset clipart --cfg configs/dt/e2e_faster_rcnn_R-101-FPN_dt-clipart-VOC.yaml --load_ckpt $dir/model_step69999.pth --multi-gpu-testing
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test_net.py --dataset clipart --cfg configs/dt/e2e_faster_rcnn_R-101-FPN_dt-clipart-VOC.yaml --load_ckpt $dir/model_step79999.pth --multi-gpu-testing
