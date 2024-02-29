#! /bin/bash

python launch.py \
    --config-name=mipnerf_dyn_humanrf_100k \
    dataset=humanrf\
    dataset.subject_id=06 \
    pos_enc=snarf \
    loss_bone_w_mult=1.0 \
    loss_bone_offset_mult=0.1