#! /bin/bash

for cid in 6 7; do
for fid in 20 160 200; do 
    ARGS_EVAL="engine=evaluator eval_cache_dir=eval_imgs_qual compute_metrics=true resume=true eval_per_gpu=1000 test_chunk=8192 eval_splits=['cam${cid}_fid${fid}_qual'] dataset.post_fix='_qual' dataset.color_bkgd='white'"
    python launch.py \
        --config-name=mipnerf_dyn_humanrf \
        dataset=humanrf \
        dataset.subject_id=03 \
        pos_enc=snarf \
        loss_bone_w_mult=1.0 \
        loss_bone_offset_mult=0.1 \
        engine=evaluator \
        $ARGS_EVAL \
        hydra.run.dir=/home/guandao/tava/pretrained/actor03
done;
done