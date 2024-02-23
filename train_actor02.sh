#! /bin/bash

CUDA_VISIBLE_DEVICES=7 python launch.py \
    --config-name=mipnerf_dyn_humanrf \
    dataset=humanrf\
    dataset.subject_id=02 \
    pos_enc=snarf \
    loss_bone_w_mult=1.0 \
    loss_bone_offset_mult=0.1