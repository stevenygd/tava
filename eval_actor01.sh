#! /bin/bash

ARGS_EVAL="engine=evaluator eval_cache_dir=eval_imgs compute_metrics=true resume=true eval_per_gpu=1000 test_chunk=10000"
python launch.py \
    --config-name=mipnerf_dyn_humanrf \
    dataset=humanrf \
    dataset.subject_id=01 \
    pos_enc=snarf \
    loss_bone_w_mult=1.0 \
    loss_bone_offset_mult=0.1 \
    engine=evaluator \
    $ARGS_EVAL \
    hydra.run.dir=/home/guandao/tava/pretrained/actor01


# ARGS_EVAL="engine=evaluator eval_cache_dir=eval_imgs compute_metrics=true resume=true eval_per_gpu=1000 test_chunk=8192 eval_splits=['cam007_fid605']"
# python launch.py \
#     --config-name=mipnerf_dyn_humanrf \
#     dataset=humanrf \
#     dataset.subject_id=01 \
#     pos_enc=snarf \
#     loss_bone_w_mult=1.0 \
#     loss_bone_offset_mult=0.1 \
#     engine=evaluator \
#     $ARGS_EVAL \
#     hydra.run.dir=/home/guandao/tava/pretrained/actor01