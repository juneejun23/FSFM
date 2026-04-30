#!/bin/bash
cd /home/junwoo/projects/FSFM/fsfm-3c/finuetune/cross_domain_FAS

PT_MODEL='/media/NAS/USERS/junwoo/FSFM_checkpoints/VF2_ViT-B/checkpoint-400.pth'

echo "=== Config ALL (MCIO 통합 학습) ==="
mkdir -p ./results/ALL
CUDA_VISIBLE_DEVICES=0 python train_vit_all.py \
    --pt_model $PT_MODEL \
    --normalize_from_IMN \
    --op_dir "./results/ALL" \
    --report_logger_path "./results/ALL/report.log"
echo "=== Config ALL DONE ==="

echo "ALL DONE!"