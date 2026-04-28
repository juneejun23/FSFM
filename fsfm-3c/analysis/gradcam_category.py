#!/usr/bin/env python3
"""
FSFM FAS fine-tuned 모델 GradCAM++ 분석 v2
- 전처리: DLIB face detection + 30% margin crop (FSFM 논문 기준)
- detection 실패 샘플 제외
- real/deepfake: C/I/M/O 4개 checkpoint 각각 (각 3000장)
- spoof 4종: 각자 해당 checkpoint (전체)
- 개별 저장: correct/misclassified 각 최대 100장, borderline 10장
- 클래스별 mean_gradcam.png (1행×4열: 전체/정분류/오분류/borderline 평균)
- 루트 avg_gradcam_grid.png (12행×4열 전체 그리드)
"""

import os
import sys
import random
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import argparse
import torch.nn.functional as F
import dlib

sys.path.insert(0, '/home/junwoo/projects/FSFM/fsfm-3c/finuetune/cross_domain_FAS')
from fas import fas_model_fix

# ── 설정 ──────────────────────────────────────────────────
PT_MODEL  = '/media/NAS/USERS/junwoo/FSFM_checkpoints/VF2_ViT-B/checkpoint-400.pth'
SAVE_DIR  = '/home/junwoo/projects/FSFM/fsfm-3c/analysis/results/gradcam_fas_v2'
os.makedirs(SAVE_DIR, exist_ok=True)

RESULTS_DIR = '/home/junwoo/projects/FSFM/fsfm-3c/finuetune/cross_domain_FAS/results'
CKPT = {
    'C': f'{RESULTS_DIR}/C/casia_vit_0_shot_checkpoint_run_0.pth.tar',
    'I': f'{RESULTS_DIR}/I/replay_vit_0_shot_checkpoint_run_0.pth.tar',
    'M': f'{RESULTS_DIR}/M/msu_vit_0_shot_checkpoint_run_0.pth.tar',
    'O': f'{RESULTS_DIR}/O/oulu_vit_0_shot_checkpoint_run_0.pth.tar',
}

MEAN = [0.5482, 0.4234, 0.3655]
STD  = [0.2789, 0.2439, 0.2349]

MAX_SAVE        = 100
MAX_BORDERLINE  = 10
RANDOM_SEED     = 42

FF_REAL_DIR = '/media/NAS/DATASET/FSFM/FaceForensics/FaceForensics/32_frames/DS_FF++_all_cls/c23/test/real_youtube'
FF_FAKE_DIR = '/media/NAS/DATASET/FSFM/FaceForensics/FaceForensics/32_frames/DS_FF++_all_cls/c23/test/fake_four'
MAX_FF      = 3000

# (img_dir, true_label, ckpt_key, max_n)
SAMPLES = {}
for k in ['C', 'I', 'M', 'O']:
    SAMPLES[f'real_{k}']     = (FF_REAL_DIR, 1, k, MAX_FF)
    SAMPLES[f'deepfake_{k}'] = (FF_FAKE_DIR, 0, k, MAX_FF)
SAMPLES['spoof_casia']  = ('/media/NAS/DATASET/FAS_FSFM/casia/test/fake',  0, 'C', None)
SAMPLES['spoof_msu']    = ('/media/NAS/DATASET/FAS_FSFM/msu/test/fake',    0, 'M', None)
SAMPLES['spoof_replay'] = ('/media/NAS/DATASET/FAS_FSFM/replay/test/fake', 0, 'I', None)
SAMPLES['spoof_oulu']   = ('/media/NAS/DATASET/FAS_FSFM/oulu/test/fake',   0, 'O', None)

# ── dlib 초기화 ────────────────────────────────────────────
detector = dlib.get_frontal_face_detector()

# ── Wrapper ────────────────────────────────────────────────
class FASModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        classifier_out, _ = self.model(x)
        return classifier_out[:, 0, :]

# ── 모델 로드 ──────────────────────────────────────────────
def load_model(fas_ckpt):
    args = argparse.Namespace(
        pt_model=PT_MODEL,
        model='vit_base_patch16',
        scratch=False,
        normalize_from_IMN=True,
        drop_path=0.0
    )
    fas_model = fas_model_fix(args)
    ckpt = torch.load(fas_ckpt, map_location='cpu', weights_only=False)
    fas_model.load_state_dict(ckpt['state_dict'])
    print(f"  Loaded: epoch={ckpt['epoch']}, best_ACC={ckpt['best_model_ACC']:.4f}")
    fas_model.eval()
    model = FASModelWrapper(fas_model)
    model.eval()
    return model

# ── 전처리 ────────────────────────────────────────────────
def preprocess(img_path):
    img = Image.open(img_path).convert('RGB')
    img_np_full = np.array(img)

    dets = detector(img_np_full, 1)
    if len(dets) == 0:
        return None, None

    det = max(dets, key=lambda d: d.width() * d.height())
    h, w = img_np_full.shape[:2]
    bw = det.right() - det.left()
    bh = det.bottom() - det.top()
    x1 = max(0, det.left()   - int(bw * 0.3))
    y1 = max(0, det.top()    - int(bh * 0.3))
    x2 = min(w, det.right()  + int(bw * 0.3))
    y2 = min(h, det.bottom() + int(bh * 0.3))

    img_cropped = img.crop((x1, y1, x2, y2))
    img_resized = img_cropped.resize((224, 224))
    img_np = np.array(img_resized) / 255.0
    img_tensor = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1)
    for c in range(3):
        img_tensor[c] = (img_tensor[c] - MEAN[c]) / STD[c]
    return img_tensor.unsqueeze(0), img_np.astype(np.float32)

# ── reshape transform ──────────────────────────────────────
def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

# ── 예측 ──────────────────────────────────────────────────
def get_prediction(model, img_tensor):
    with torch.no_grad():
        output = model(img_tensor)
        prob = F.softmax(output, dim=1)[0]
        pred_class = prob.argmax().item()
        prob_real = prob[1].item()
        prob_fake = prob[0].item()
    return pred_class, prob_real, prob_fake

# ── 개별 GradCAM++ 저장 ────────────────────────────────────
def save_gradcam(model, img_tensor, img_np, pred_class,
                 prob_real, prob_fake, true_label, save_path, fname):
    target_layers = [model.model.backbone.vit.blocks[-1].norm1]
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers,
                          reshape_transform=reshape_transform)
    targets = [ClassifierOutputTarget(pred_class)]
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0]
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    correct = '✓' if pred_class == true_label else '✗'
    pred_str = 'real' if pred_class == 1 else 'fake'
    true_str = 'real' if true_label == 1 else 'fake'

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(
        f'{fname}\ntrue={true_str} | pred={pred_str} {correct} | '
        f'P(real)={prob_real:.3f} P(fake)={prob_fake:.3f}', fontsize=9)
    axes[0].imshow((img_np * 255).astype(np.uint8))
    axes[0].set_title('Original'); axes[0].axis('off')
    im = axes[1].imshow(grayscale_cam, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title('GradCAM++ heatmap'); axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    axes[2].imshow(visualization)
    axes[2].set_title('Overlay'); axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ── GradCAM 맵 리스트 계산 ─────────────────────────────────
def compute_cam_maps(model, preds):
    if not preds:
        return []
    target_layers = [model.model.backbone.vit.blocks[-1].norm1]
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers,
                          reshape_transform=reshape_transform)
    cam_maps = []
    for p in preds:
        targets = [ClassifierOutputTarget(p['pred_class'])]
        grayscale_cam = cam(input_tensor=p['img_tensor'], targets=targets)[0]
        cam_maps.append(grayscale_cam)
    return cam_maps

# ── 클래스별 mean_gradcam.png (1행×4열) ───────────────────
def save_cls_mean_gradcam(all_maps, correct_maps, wrong_maps, borderline_maps,
                          cls_name, save_path):
    col_titles = ['All Mean', 'Correct Mean', 'Misclassified Mean', 'Borderline Mean']
    maps_list  = [all_maps, correct_maps, wrong_maps, borderline_maps]

    fig, axes = plt.subplots(1, 4, figsize=(3.8 * 4, 4))
    for col_idx, (maps, title) in enumerate(zip(maps_list, col_titles)):
        ax = axes[col_idx]
        ax.set_title(title, fontsize=10, fontweight='bold')
        if maps:
            avg_map = np.mean(maps, axis=0)
            im = ax.imshow(avg_map, cmap='jet', vmin=0, vmax=1)
            ax.set_xlabel(f'n={len(maps)}', fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.text(0.5, 0.5, 'No samples', ha='center', va='center',
                    fontsize=10, color='gray', transform=ax.transAxes)
            ax.set_facecolor('#eeeeee')
        ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle(f'{cls_name} — Average GradCAM++', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ── 루트 avg_gradcam_grid.png (12행×4열) ──────────────────
def plot_avg_gradcam_grid(all_class_data, save_path):
    n_rows = len(all_class_data)
    n_cols = 4
    col_titles = ['All Mean', 'Correct Mean', 'Misclassified Mean', 'Borderline Mean']

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.8, n_rows * 3.8))
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=13, fontweight='bold', pad=12)

    for row_idx, cls_data in enumerate(all_class_data):
        cls_name    = cls_data['cls_name']
        maps_by_col = [
            cls_data['all_maps'], cls_data['correct_maps'],
            cls_data['wrong_maps'], cls_data['borderline_maps']
        ]
        axes[row_idx, 0].set_ylabel(cls_name, fontsize=10, fontweight='bold', labelpad=12)

        for col_idx, maps in enumerate(maps_by_col):
            ax = axes[row_idx, col_idx]
            if maps:
                avg_map = np.mean(maps, axis=0)
                im = ax.imshow(avg_map, cmap='jet', vmin=0, vmax=1)
                ax.set_xlabel(f'n={len(maps)}', fontsize=9)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.text(0.5, 0.5, 'No samples', ha='center', va='center',
                        fontsize=11, color='gray', transform=ax.transAxes)
                ax.set_facecolor('#eeeeee')
            ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle(
        'FAS fine-tuned Model: Average GradCAM++ Map (DLIB crop)\n'
        '(All Mean / Correct Mean / Misclassified Mean / Borderline Mean)',
        fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 전체 그리드 저장: {save_path}")

# ── 메인 ──────────────────────────────────────────────────
if __name__ == '__main__':
    random.seed(RANDOM_SEED)

    all_results    = []
    all_class_data = []
    loaded_ckpts   = {}

    for cls_name, (img_dir, true_label, ckpt_key, max_n) in SAMPLES.items():
        print(f"\n=== {cls_name} (ckpt={ckpt_key}) ===")
        if not os.path.exists(img_dir):
            print(f"  Path not found: {img_dir}"); continue

        all_files = sorted([
            f for f in os.listdir(img_dir)
            if f.endswith('.png') or f.endswith('.jpg')
        ])
        if max_n and len(all_files) > max_n:
            all_files = random.sample(all_files, max_n)

        fas_ckpt = CKPT[ckpt_key]
        if fas_ckpt not in loaded_ckpts:
            print(f"  Loading checkpoint: {fas_ckpt}")
            loaded_ckpts[fas_ckpt] = load_model(fas_ckpt).cuda()
        model = loaded_ckpts[fas_ckpt]

        for sub in ['correct', 'misclassified', 'borderline']:
            os.makedirs(os.path.join(SAVE_DIR, cls_name, sub), exist_ok=True)
        npy_dir = os.path.join(SAVE_DIR, cls_name, 'maps_npy')
        os.makedirs(npy_dir, exist_ok=True)

        # Step 1: 전체 예측 (detection 실패 제외)
        preds  = []
        n_skip = 0
        for fname in all_files:
            img_path = os.path.join(img_dir, fname)
            img_tensor, img_np = preprocess(img_path)
            if img_tensor is None:
                n_skip += 1
                continue
            img_tensor = img_tensor.cuda()
            pred_class, prob_real, prob_fake = get_prediction(model, img_tensor)
            preds.append({
                'fname':      fname,
                'img_tensor': img_tensor,
                'img_np':     img_np,
                'pred_class': pred_class,
                'prob_real':  prob_real,
                'prob_fake':  prob_fake,
                'correct':    pred_class == true_label,
            })

        total     = len(preds)
        n_correct = sum(1 for p in preds if p['correct'])
        n_wrong   = total - n_correct
        print(f"  Skip={n_skip} | Valid={total} | Correct={n_correct} ({n_correct/total*100:.1f}%) | Misclassified={n_wrong} ({n_wrong/total*100:.1f}%)")

        all_results.append({
            'class': cls_name, 'ckpt': ckpt_key, 'skipped': n_skip,
            'total': total, 'correct': n_correct, 'misclassified': n_wrong,
            'accuracy': f'{n_correct/total*100:.1f}%',
            'miss_rate': f'{n_wrong/total*100:.1f}%',
        })

        # Step 2: 분류
        wrong_preds      = [p for p in preds if not p['correct']]
        correct_preds    = [p for p in preds if p['correct']]
        borderline_preds = sorted(preds, key=lambda x: abs(x['prob_fake'] - 0.5))[:MAX_BORDERLINE]

        wrong_save   = random.sample(wrong_preds,   min(MAX_SAVE, len(wrong_preds)))
        correct_save = random.sample(correct_preds, min(MAX_SAVE, len(correct_preds)))

        # Step 3: 개별 GradCAM 저장
        for i, p in enumerate(correct_save):
            save_path = os.path.join(SAVE_DIR, cls_name, 'correct', f'{i:02d}.png')
            save_gradcam(model, p['img_tensor'], p['img_np'], p['pred_class'],
                         p['prob_real'], p['prob_fake'], true_label, save_path, p['fname'])
        print(f"  Correct 저장: {len(correct_save)}장")

        for i, p in enumerate(wrong_save):
            save_path = os.path.join(SAVE_DIR, cls_name, 'misclassified', f'{i:02d}.png')
            save_gradcam(model, p['img_tensor'], p['img_np'], p['pred_class'],
                         p['prob_real'], p['prob_fake'], true_label, save_path, p['fname'])
        print(f"  Misclassified 저장: {len(wrong_save)}장")

        for i, p in enumerate(borderline_preds):
            save_path = os.path.join(SAVE_DIR, cls_name, 'borderline', f'{i:02d}.png')
            save_gradcam(model, p['img_tensor'], p['img_np'], p['pred_class'],
                         p['prob_real'], p['prob_fake'], true_label, save_path, p['fname'])
        print(f"  Borderline 저장: {len(borderline_preds)}장")

        # Step 4: 평균 GradCAM 맵 계산
        print(f"  Computing mean GradCAM maps (전체 {total}장)...")
        all_maps        = compute_cam_maps(model, preds)
        correct_maps    = compute_cam_maps(model, correct_preds)
        wrong_maps      = compute_cam_maps(model, wrong_preds)
        borderline_maps = compute_cam_maps(model, borderline_preds)

        if all_maps:
            np.save(os.path.join(npy_dir, 'mean_all.npy'),        np.mean(all_maps,        axis=0))
        if correct_maps:
            np.save(os.path.join(npy_dir, 'mean_correct.npy'),    np.mean(correct_maps,    axis=0))
        if wrong_maps:
            np.save(os.path.join(npy_dir, 'mean_wrong.npy'),      np.mean(wrong_maps,      axis=0))
        if borderline_maps:
            np.save(os.path.join(npy_dir, 'mean_borderline.npy'), np.mean(borderline_maps, axis=0))

        # 클래스별 mean_gradcam.png (1행×4열)
        cls_mean_path = os.path.join(SAVE_DIR, cls_name, 'mean_gradcam.png')
        save_cls_mean_gradcam(all_maps, correct_maps, wrong_maps, borderline_maps,
                              cls_name, cls_mean_path)
        print(f"  mean_gradcam.png 저장")

        all_class_data.append({
            'cls_name':       cls_name,
            'all_maps':       all_maps,
            'correct_maps':   correct_maps,
            'wrong_maps':     wrong_maps,
            'borderline_maps': borderline_maps,
        })
        print(f"  Maps: 전체={len(all_maps)} / Correct={len(correct_maps)} / Misclassified={len(wrong_maps)} / Borderline={len(borderline_maps)}")

    # ── 통계 표 저장 ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, len(all_results) * 0.5 + 1.5))
    ax.axis('off')
    col_labels = ['class', 'ckpt', 'skipped', 'total', 'correct', 'misclassified', 'accuracy', 'miss_rate']
    table_data = [[r[c] for c in col_labels] for r in all_results]
    table = ax.table(cellText=table_data, colLabels=col_labels, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(col_labels))))
    for row_idx, r in enumerate(all_results):
        miss = float(r['miss_rate'].replace('%', ''))
        color = '#ffcccc' if miss > 20 else '#ffffcc' if miss > 5 else '#ccffcc'
        for col_idx in range(len(col_labels)):
            table[row_idx + 1, col_idx].set_facecolor(color)
    plt.title('FAS Model - Classification Statistics (DLIB crop, v2)', fontsize=12, pad=10)
    plt.tight_layout()
    table_path = os.path.join(SAVE_DIR, 'statistics_table.png')
    plt.savefig(table_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n통계 표 저장: {table_path}")

    # ── 루트 avg_gradcam_grid.png (12행×4열) ──────────────
    grid_path = os.path.join(SAVE_DIR, 'avg_gradcam_grid.png')
    plot_avg_gradcam_grid(all_class_data, grid_path)

    print(f"\n✅ 전체 완료!")
    print(f"   통계 표:     {table_path}")
    print(f"   전체 그리드: {grid_path}")