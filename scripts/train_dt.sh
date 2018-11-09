#!/usr/bin/env bash#!/usr/bin/env bash
python tools/train_net_step.py --cfg configs/dt/e2e_faster_rcnn_R-101-FPN_dt-clipart-VOC.yaml --load_ckpt Outputs/e2e_faster_rcnn_R-101-FPN_1x-VOC/Oct08-14-54-41_ip-172-31-8-158_step/ckpt/model_step9999.pth --use_tfboard --dataset dt_clipart_voc0712 --bs 12   2>&1|tee train_dt_clipart_VOC-lr_1e-4.log
