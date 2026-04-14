#!/bin/bash
cd /home/junwoo/projects/FSFM/fsfm-3c/finuetune/cross_domain_FAS

PT_MODEL='/media/NAS/USERS/junwoo/FSFM_checkpoints/VF2_ViT-B/checkpoint-400.pth'

for config in C I M O; do
    echo "=== Config $config ==="
    mkdir -p ./results/${config}
    CUDA_VISIBLE_DEVICES=0 python train_vit.py \
        --config $config \
        --pt_model $PT_MODEL \
        --normalize_from_IMN \
        --op_dir "./results/${config}" \
        --report_logger_path "./results/${config}/report.log"
    echo "=== Config $config DONE ==="
done

echo "ALL DONE!"
