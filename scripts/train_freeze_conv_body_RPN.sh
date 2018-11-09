#!/bin/bash
python tools/train_net_step.py --cfg configs/dt/e2e_faster_rcnn_R-101-FPN_clipart_weakly_without-spatial-reg_copy-cls-to-det_freeze-conv-body-RPN.yaml --load_ckpt Outputs/e2e_faster_rcnn_R-101-FPN_1x-VOC/Oct08-14-54-41_ip-172-31-8-158_step/ckpt/model_step9999.pth --use_tfboard --dataset clipart 2>&1|tee log/train_no_reg_copy_det_freeze_conv_RPN.output.2e-4.1108.log

python tools/train_net_step.py --cfg configs/dt/e2e_faster_rcnn_R-101-FPN_clipart_weakly_with-spatial-reg_copy-cls-to-det_freeze-conv-body-RPN.yaml --load_ckpt Outputs/e2e_faster_rcnn_R-101-FPN_1x-VOC/Oct08-14-54-41_ip-172-31-8-158_step/ckpt/model_step9999.pth --use_tfboard --dataset clipart 2>&1|tee log/train_reg_copy_det_freeze_conv_RPN.output.2e-4.1108.log
