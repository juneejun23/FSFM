#!/usr/bin/env python3
"""
FSFM FAS fine-tuned 모델 - Spoof 4종 카테고리별 GradCAM++ 분석
- 전처리: DLIB face detection + 30% margin crop (FSFM 논문 기준)
- spoof 4종 각각: fake + real 폴더 합쳐서 처리
- 각 데이터셋 폴더 안에 카테고리별 폴더 생성:
    TP_high_conf  : gt=fake, pred=fake, P(fake) >= 0.85
    TN_high_conf  : gt=real, pred=real, P(fake) <= 0.15
    FP_false_alarm: gt=real, pred=fake
    FN_missed     : gt=fake, pred=real
    borderline    : P(fake) 0.5에 가장 가까운 것
- 카테고리별 최대 50장 저장 + 평균 GradCAM 맵
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
SAVE_DIR  = '/home/junwoo/projects/FSFM/fsfm-3c/analysis/results/gradcam_spoof_category'
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

MAX_SAVE    = 50
RANDOM_SEED = 42

# (fake_dir, real_dir, ckpt_key)
SAMPLES = {
    'spoof_casia':  (
        '/media/NAS/DATASET/FAS_FSFM/casia/test/fake',
        '/media/NAS/DATASET/FAS_FSFM/casia/test/real',
        'C'
    ),
    'spoof_msu':    (
        '/media/NAS/DATASET/FAS_FSFM/msu/test/fake',
        '/media/NAS/DATASET/FAS_FSFM/msu/test/real',
        'M'
    ),
    'spoof_replay': (
        '/media/NAS/DATASET/FAS_FSFM/replay/test/fake',
        '/media/NAS/DATASET/FAS_FSFM/replay/test/real',
        'I'
    ),
    'spoof_oulu':   (
        '/media/NAS/DATASET/FAS_FSFM/oulu/test/fake',
        '/media/NAS/DATASET/FAS_FSFM/oulu/test/real',
        'O'
    ),
}

CATEGORIES = ['TP_high_conf', 'TN_high_conf', 'FP_false_alarm', 'FN_missed', 'borderline']

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

# ── 카테고리 분류 ──────────────────────────────────────────
def classify_category(true_label, pred_class, prob_fake):
    if true_label == 0 and pred_class == 0 and prob_fake >= 0.85:
        return 'TP_high_conf'
    elif true_label == 1 and pred_class == 1 and prob_fake <= 0.15:
        return 'TN_high_conf'
    elif true_label == 1 and pred_class == 0:
        return 'FP_false_alarm'
    elif true_label == 0 and pred_class == 1:
        return 'FN_missed'
    else:
        return 'other'

# ── GradCAM 저장 ──────────────────────────────────────────
def save_gradcam(model, img_tensor, img_np, pred_class,
                 prob_real, prob_fake, true_label, category, save_path, fname):
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
        f'[{category}] {fname}\n'
        f'true={true_str} | pred={pred_str} {correct} | '
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
    return grayscale_cam

# ── 카테고리 평균 GradCAM 저장 ────────────────────────────
def save_category_mean(cam_maps, category, cls_name, save_path):
    if not cam_maps:
        return
    avg_map = np.mean(cam_maps, axis=0)
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    im = axes.imshow(avg_map, cmap='jet', vmin=0, vmax=1)
    axes.set_title(f'{cls_name} / {category}\nMean GradCAM++ (n={len(cam_maps)})',
                   fontsize=11, fontweight='bold')
    axes.axis('off')
    plt.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ── 메인 ──────────────────────────────────────────────────
if __name__ == '__main__':
    random.seed(RANDOM_SEED)
    loaded_ckpts = {}
    summary = []

    for cls_name, (fake_dir, real_dir, ckpt_key) in SAMPLES.items():
        print(f"\n=== {cls_name} (ckpt={ckpt_key}) ===")

        # checkpoint 로드 (캐싱)
        fas_ckpt = CKPT[ckpt_key]
        if fas_ckpt not in loaded_ckpts:
            print(f"  Loading checkpoint: {fas_ckpt}")
            loaded_ckpts[fas_ckpt] = load_model(fas_ckpt).cuda()
        model = loaded_ckpts[fas_ckpt]

        # 카테고리 폴더 생성
        for cat in CATEGORIES:
            os.makedirs(os.path.join(SAVE_DIR, cls_name, cat), exist_ok=True)

        # fake + real 파일 수집
        all_items = []  # (img_path, true_label)
        for img_dir, true_label in [(fake_dir, 0), (real_dir, 1)]:
            if not os.path.exists(img_dir):
                print(f"  Path not found: {img_dir}"); continue
            files = sorted([
                f for f in os.listdir(img_dir)
                if f.endswith('.png') or f.endswith('.jpg')
            ])
            for f in files:
                all_items.append((os.path.join(img_dir, f), true_label))

        random.shuffle(all_items)

        # 전체 예측 + 카테고리 분류
        buckets = {cat: [] for cat in CATEGORIES}
        borderline_all = []
        n_skip = 0

        for img_path, true_label in all_items:
            fname = os.path.basename(img_path)
            img_tensor, img_np = preprocess(img_path)
            if img_tensor is None:
                n_skip += 1
                continue
            img_tensor = img_tensor.cuda()
            pred_class, prob_real, prob_fake = get_prediction(model, img_tensor)

            entry = {
                'fname': fname, 'img_path': img_path,
                'img_tensor': img_tensor, 'img_np': img_np,
                'pred_class': pred_class, 'prob_real': prob_real,
                'prob_fake': prob_fake, 'true_label': true_label,
                'borderline_dist': abs(prob_fake - 0.5),
            }

            cat = classify_category(true_label, pred_class, prob_fake)
            if cat != 'other':
                buckets[cat].append(entry)
            borderline_all.append(entry)

        # borderline: 0.5에 가장 가까운 순
        buckets['borderline'] = sorted(borderline_all, key=lambda x: x['borderline_dist'])

        total_valid = len(all_items) - n_skip
        print(f"  Skip={n_skip} | Valid={total_valid}")
        for cat in CATEGORIES:
            print(f"  {cat}: {len(buckets[cat])}개 후보")

        # 카테고리별 GradCAM 저장 + 평균 맵
        cls_summary = {'class': cls_name, 'ckpt': ckpt_key}
        for cat in CATEGORIES:
            candidates = buckets[cat]
            if cat == 'borderline':
                selected = candidates[:MAX_SAVE]
            else:
                selected = random.sample(candidates, min(MAX_SAVE, len(candidates)))

            cam_maps = []
            for i, p in enumerate(selected):
                save_path = os.path.join(SAVE_DIR, cls_name, cat, f'{i:02d}.png')
                cam = save_gradcam(
                    model, p['img_tensor'], p['img_np'], p['pred_class'],
                    p['prob_real'], p['prob_fake'], p['true_label'],
                    cat, save_path, p['fname']
                )
                cam_maps.append(cam)

            # 카테고리 평균 GradCAM
            mean_path = os.path.join(SAVE_DIR, cls_name, cat, 'mean_gradcam.png')
            save_category_mean(cam_maps, cat, cls_name, mean_path)

            cls_summary[cat] = len(selected)
            print(f"  {cat}: {len(selected)}장 저장")

        summary.append(cls_summary)

    # ── 전체 요약 출력 ──────────────────────────────────────
    print("\n\n=== 전체 요약 ===")
    print(f"{'class':<20} {'ckpt':<6} " + " ".join(f"{c:<16}" for c in CATEGORIES))
    print("-" * 110)
    for s in summary:
        row = f"{s['class']:<20} {s['ckpt']:<6} "
        row += " ".join(f"{s.get(c, 0):<16}" for c in CATEGORIES)
        print(row)

    print(f"\n✅ 완료! 저장 경로: {SAVE_DIR}")