#!/bin/bash
dir=Outputs/e2e_faster_rcnn_R-101-FPN_dt-clipart-VOC/Nov04-20-33-03_ip-172-31-8-158_step/ckpt

python tools/train_net_step.py --cfg configs/dt/e2e_faster_rcnn_R-101-FPN_clipart_weakly_1stream_noreg_freeze-boxhead_unlock-rpn-conv.yaml --load_ckpt $dir/model_step9999.pth --use_tfboard --dataset clipart --bs 32 2>&1|tee log/train_1stream_noreg_freeze-boxhead_unlock-rpn-conv.from_dt.2e-5.1114.log

python tools/train_net_step.py --cfg configs/dt/e2e_faster_rcnn_R-101-FPN_clipart_weakly_1stream_noreg_freeze-boxhead-conv.yaml --load_ckpt $dir/model_step9999.pth --use_tfboard --dataset clipart --bs 32 2>&1|tee log/train_1stream_noreg_freeze-boxhead-conv.from_dt.output.2e-5.1114.log
