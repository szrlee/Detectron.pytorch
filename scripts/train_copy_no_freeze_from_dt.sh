#!/bin/bash
dir=Outputs/e2e_faster_rcnn_R-101-FPN_dt-clipart-VOC/Nov04-20-33-03_ip-172-31-8-158_step/ckpt

python tools/train_net_step.py --cfg configs/dt/e2e_faster_rcnn_R-101-FPN_clipart_weakly_without-spatial-reg_copy-cls-to-det.yaml --load_ckpt $dir/model_step9999.pth --use_tfboard --dataset clipart 2>&1|tee train_no_reg_copy_det_no_freeze.from_dt.output.2e-4.1109.log

python tools/train_net_step.py --cfg configs/dt/e2e_faster_rcnn_R-101-FPN_clipart_weakly_with-spatial-reg_copy-cls-to-det.yaml --load_ckpt $dir/model_step9999.pth --use_tfboard --dataset clipart 2>&1|tee train_reg_copy_det_no_freeze.from_dt.output.2e-4.1109.log
