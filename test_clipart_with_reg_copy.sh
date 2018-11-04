#!/usr/bin/env bash
dir=Outputs/e2e_faster_rcnn_R-101-FPN_clipart_weakly_with-spatial-reg_copy-cls-to-det/Oct29-16-20-26_ip-172-31-8-158_step/ckpt

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test_net.py --dataset clipart --cfg configs/dt/e2e_faster_rcnn_R-101-FPN_clipart_weakly_with-spatial-reg_copy-cls-to-det.yaml --load_ckpt $dir/model_step499.pth --multi-gpu-testing
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test_net.py --dataset clipart --cfg configs/dt/e2e_faster_rcnn_R-101-FPN_clipart_weakly_with-spatial-reg_copy-cls-to-det.yaml --load_ckpt $dir/model_step999.pth --multi-gpu-testing
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test_net.py --dataset clipart --cfg configs/dt/e2e_faster_rcnn_R-101-FPN_clipart_weakly_with-spatial-reg_copy-cls-to-det.yaml --load_ckpt $dir/model_step1499.pth --multi-gpu-testing
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test_net.py --dataset clipart --cfg configs/dt/e2e_faster_rcnn_R-101-FPN_clipart_weakly_with-spatial-reg_copy-cls-to-det.yaml --load_ckpt $dir/model_step1999.pth --multi-gpu-testing
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test_net.py --dataset clipart --cfg configs/dt/e2e_faster_rcnn_R-101-FPN_clipart_weakly_with-spatial-reg_copy-cls-to-det.yaml --load_ckpt $dir/model_step2499.pth --multi-gpu-testing
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test_net.py --dataset clipart --cfg configs/dt/e2e_faster_rcnn_R-101-FPN_clipart_weakly_with-spatial-reg_copy-cls-to-det.yaml --load_ckpt $dir/model_step2999.pth --multi-gpu-testing
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test_net.py --dataset clipart --cfg configs/dt/e2e_faster_rcnn_R-101-FPN_clipart_weakly_with-spatial-reg_copy-cls-to-det.yaml --load_ckpt $dir/model_step3499.pth --multi-gpu-testing
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test_net.py --dataset clipart --cfg configs/dt/e2e_faster_rcnn_R-101-FPN_clipart_weakly_with-spatial-reg_copy-cls-to-det.yaml --load_ckpt $dir/model_step3999.pth --multi-gpu-testing
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test_net.py --dataset clipart --cfg configs/dt/e2e_faster_rcnn_R-101-FPN_clipart_weakly_with-spatial-reg_copy-cls-to-det.yaml --load_ckpt $dir/model_step4499.pth --multi-gpu-testing
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test_net.py --dataset clipart --cfg configs/dt/e2e_faster_rcnn_R-101-FPN_clipart_weakly_with-spatial-reg_copy-cls-to-det.yaml --load_ckpt $dir/model_step4999.pth --multi-gpu-testing
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test_net.py --dataset clipart --cfg configs/dt/e2e_faster_rcnn_R-101-FPN_clipart_weakly_with-spatial-reg_copy-cls-to-det.yaml --load_ckpt $dir/model_step5499.pth --multi-gpu-testing
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test_net.py --dataset clipart --cfg configs/dt/e2e_faster_rcnn_R-101-FPN_clipart_weakly_with-spatial-reg_copy-cls-to-det.yaml --load_ckpt $dir/model_step5999.pth --multi-gpu-testing
