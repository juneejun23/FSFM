#!/usr/bin/env python3
"""
각 클래스/카테고리 폴더 안의 이미지 10장을 2행×5열 그리드로 저장
- real/deepfake: gradcam_fas_v2 → correct/misclassified/borderline
- spoof 4종: gradcam_spoof_category → TP_high_conf/TN_high_conf/FP_false_alarm/FN_missed/borderline
- 새 폴더 gradcam_grid_overview에 동일 구조로 grid_overview.png 저장
"""

import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ── 경로 설정 ──────────────────────────────────────────────
V2_DIR       = '/home/junwoo/projects/FSFM/fsfm-3c/analysis/results/gradcam_fas_v2'
SPOOF_DIR    = '/home/junwoo/projects/FSFM/fsfm-3c/analysis/results/gradcam_spoof_category'
OUT_DIR      = '/home/junwoo/projects/FSFM/fsfm-3c/analysis/results/gradcam_grid_overview'

FF_CLASSES    = [f'{t}_{k}' for t in ['real', 'deepfake'] for k in ['C', 'I', 'M', 'O']]
SPOOF_CLASSES = ['spoof_casia', 'spoof_msu', 'spoof_replay', 'spoof_oulu']

FF_CATEGORIES    = ['correct', 'misclassified', 'borderline']
SPOOF_CATEGORIES = ['TP_high_conf', 'TN_high_conf', 'FP_false_alarm', 'FN_missed', 'borderline']

N_ROWS = 10
N_COLS = 1

# ── 그리드 생성 함수 ───────────────────────────────────────
def make_grid(src_folder, dst_folder, cls_name, cat_name):
    os.makedirs(dst_folder, exist_ok=True)

    imgs = sorted([
        f for f in os.listdir(src_folder)
        if f.endswith('.png')
        and f not in ('grid_overview.png', 'mean_gradcam.png')
    ])

    if len(imgs) == 0:
        print(f"  [SKIP] 이미지 없음: {src_folder}")
        return

    selected = random.sample(imgs, min(N_ROWS * N_COLS, len(imgs)))

    loaded = []
    for fname in selected:
        try:
            img = Image.open(os.path.join(src_folder, fname)).convert('RGB')
            loaded.append((fname, np.array(img)))
        except Exception as e:
            print(f"  [WARN] 로드 실패: {fname} - {e}")

    if not loaded:
        return

    n_total      = len(loaded)
    actual_rows  = (n_total + N_COLS - 1) // N_COLS

    fig, axes = plt.subplots(actual_rows, N_COLS,
                              figsize=(N_COLS * 5, actual_rows * 2.2))

    if actual_rows == 1:
        axes = [axes]
    axes_flat = [ax for row in axes for ax in (row if hasattr(row, '__iter__') else [row])]

    for idx, ax in enumerate(axes_flat):
        if idx < len(loaded):
            fname, img_arr = loaded[idx]
            ax.imshow(img_arr)
            ax.set_title(fname, fontsize=5, pad=2)
        ax.axis('off')

    plt.suptitle(f'{cls_name} / {cat_name}  (n={len(loaded)})',
                 fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout()

    save_path = os.path.join(dst_folder, 'grid_overview.png')
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {save_path}")

# ── 메인 ──────────────────────────────────────────────────
if __name__ == '__main__':

    # ── real/deepfake (gradcam_fas_v2) ─────────────────────
    print("\n=== real/deepfake (gradcam_fas_v2) ===")
    for cls_name in FF_CLASSES:
        src_cls = os.path.join(V2_DIR, cls_name)
        dst_cls = os.path.join(OUT_DIR, cls_name)
        if not os.path.isdir(src_cls):
            print(f"  [SKIP] {cls_name} 없음")
            continue
        print(f"\n[{cls_name}]")
        for cat in FF_CATEGORIES:
            src_cat = os.path.join(src_cls, cat)
            dst_cat = os.path.join(dst_cls, cat)
            if os.path.isdir(src_cat):
                make_grid(src_cat, dst_cat, cls_name, cat)

    # ── spoof 4종 (gradcam_spoof_category) ─────────────────
    print("\n=== spoof 4종 (gradcam_spoof_category) ===")
    for cls_name in SPOOF_CLASSES:
        src_cls = os.path.join(SPOOF_DIR, cls_name)
        dst_cls = os.path.join(OUT_DIR, cls_name)
        if not os.path.isdir(src_cls):
            print(f"  [SKIP] {cls_name} 없음")
            continue
        print(f"\n[{cls_name}]")
        for cat in SPOOF_CATEGORIES:
            src_cat = os.path.join(src_cls, cat)
            dst_cat = os.path.join(dst_cls, cat)
            if os.path.isdir(src_cat):
                make_grid(src_cat, dst_cat, cls_name, cat)

    print(f"\n✅ 전체 완료! 저장 경로: {OUT_DIR}")