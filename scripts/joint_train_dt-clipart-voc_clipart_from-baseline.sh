#!/bin/bash
pretrain=Outputs/e2e_faster_rcnn_R-101-FPN_1x-VOC/Oct08-14-54-41_ip-172-31-8-158_step/ckpt/model_step9999.pth
config=configs/joint/e2e_faster_rcnn_R-101-FPN_JointTrain_dt-clipart-voc_clipart_pseudo.yaml
dataset=dt_clipart_voc0712+clipart
python tools/train_net_step.py --cfg $config --load_ckpt $pretrain --use_tfboard --dataset $dataset --bs 24 2>&1|tee log/joint_train_dt-clipart-voc_clipart.from_baseline.4e-5.1115.log
