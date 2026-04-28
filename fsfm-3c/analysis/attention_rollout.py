#!/usr/bin/env python3
"""
FSFM FAS fine-tuned 모델 Attention Rollout 분석
- label: 0=fake/spoof, 1=real/live
- 전처리: DLIB face detection + 30% margin crop (FSFM 논문 기준)
- detection 실패 샘플 제외
- FF++ real/deepfake: 각 3000장 제한 / FAS 4종: 전체
- 평균 Attention Map: 전체 샘플 기준
- 개별 저장: 최대 100장 랜덤 샘플링
"""

import os
import sys
import random
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import argparse
import dlib

sys.path.insert(0, '/home/junwoo/projects/FSFM/fsfm-3c/finuetune/cross_domain_FAS')
from fas import fas_model_fix

# ── 설정 ──────────────────────────────────────────────────
PT_MODEL  = '/media/NAS/USERS/junwoo/FSFM_checkpoints/VF2_ViT-B/checkpoint-400.pth'
SAVE_DIR  = '/home/junwoo/projects/FSFM/fsfm-3c/analysis/results/attention_rollout_fas_dlib'
os.makedirs(SAVE_DIR, exist_ok=True)

RESULTS_DIR = '/home/junwoo/projects/FSFM/fsfm-3c/finuetune/cross_domain_FAS/results'
FAS_CKPTS = {
    'real':         f'{RESULTS_DIR}/O/oulu_vit_0_shot_checkpoint_run_0.pth.tar',
    'deepfake':     f'{RESULTS_DIR}/O/oulu_vit_0_shot_checkpoint_run_0.pth.tar',
    'spoof_casia':  f'{RESULTS_DIR}/C/casia_vit_0_shot_checkpoint_run_0.pth.tar',
    'spoof_msu':    f'{RESULTS_DIR}/M/msu_vit_0_shot_checkpoint_run_0.pth.tar',
    'spoof_replay': f'{RESULTS_DIR}/I/replay_vit_0_shot_checkpoint_run_0.pth.tar',
    'spoof_oulu':   f'{RESULTS_DIR}/O/oulu_vit_0_shot_checkpoint_run_0.pth.tar',
}

MEAN = [0.5482, 0.4234, 0.3655]
STD  = [0.2789, 0.2439, 0.2349]

MAX_SAVE    = 100
RANDOM_SEED = 42

# (경로, true_label, 최대샘플수) - None이면 전체
SAMPLES = {
    'real':         ('/media/NAS/DATASET/FSFM/FaceForensics/FaceForensics/32_frames/DS_FF++_all_cls/c23/test/real_youtube', 1, 3000),
    'deepfake':     ('/media/NAS/DATASET/FSFM/FaceForensics/FaceForensics/32_frames/DS_FF++_all_cls/c23/test/fake_four',    0, 3000),
    'spoof_casia':  ('/media/NAS/DATASET/FAS_FSFM/casia/test/fake',  0, None),
    'spoof_msu':    ('/media/NAS/DATASET/FAS_FSFM/msu/test/fake',    0, None),
    'spoof_replay': ('/media/NAS/DATASET/FAS_FSFM/replay/test/fake', 0, None),
    'spoof_oulu':   ('/media/NAS/DATASET/FAS_FSFM/oulu/test/fake',   0, None),
}

# ── dlib 초기화 ────────────────────────────────────────────
detector = dlib.get_frontal_face_detector()

# ── Wrapper ────────────────────────────────────────────────
class FASWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out, _ = self.model(x)
        return out[:, 0, :]

# ── Attention Rollout ──────────────────────────────────────
class AttentionRollout:
    def __init__(self, model, discard_ratio=0.9):
        self.model = model
        self.discard_ratio = discard_ratio
        self.attentions = []
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def hook_fn(module, input, output):
            self.attentions.append(input[0].detach().cpu())
        for blk in self.model.model.backbone.vit.blocks:
            h = blk.attn.attn_drop.register_forward_hook(hook_fn)
            self.hooks.append(h)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()

    def __call__(self, img_tensor):
        self.attentions = []
        with torch.no_grad():
            _ = self.model(img_tensor)

        result = torch.eye(self.attentions[0].shape[-1])
        for attn in self.attentions:
            attn = attn[0].mean(dim=0)
            flat = attn.view(-1)
            threshold = flat.kthvalue(int(flat.size(0) * self.discard_ratio)).values
            attn = attn * (attn > threshold).float()
            attn = attn + torch.eye(attn.shape[0])
            attn = attn / attn.sum(dim=-1, keepdim=True)
            result = torch.matmul(attn, result)

        mask = result[0, 1:].reshape(14, 14)
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        return mask.numpy()

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
    print(f"Loaded FAS checkpoint: epoch={ckpt['epoch']}, best_ACC={ckpt['best_model_ACC']:.4f}")
    fas_model.eval()
    model = FASWrapper(fas_model)
    model.eval()
    return model

# ── 전처리 ────────────────────────────────────────────────
def preprocess(img_path):
    img = Image.open(img_path).convert('RGB')
    img_np_full = np.array(img)

    dets = detector(img_np_full, 1)
    if len(dets) == 0:
        return None, None  # detection 실패 → 제외

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

# ── 예측 ──────────────────────────────────────────────────
def get_prediction(model, img_tensor):
    with torch.no_grad():
        output = model(img_tensor)
        prob = F.softmax(output, dim=1)[0]
        pred_class = prob.argmax().item()
        prob_real = prob[1].item()
        prob_fake = prob[0].item()
    return pred_class, prob_real, prob_fake

# ── 개별 Attention Rollout 저장 ────────────────────────────
def save_attention(img_np, mask, pred_class, prob_real, prob_fake,
                   true_label, save_path, fname):
    mask_resized = np.array(
        Image.fromarray((mask * 255).astype(np.uint8)).resize((224, 224), Image.BICUBIC)
    ) / 255.0
    heatmap = plt.cm.jet(mask_resized)[:, :, :3]
    overlay = np.clip(0.5 * img_np + 0.5 * heatmap, 0, 1)

    correct = '✓' if pred_class == true_label else '✗'
    pred_str = 'real' if pred_class == 1 else 'fake'
    true_str = 'real' if true_label == 1 else 'fake'

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(
        f'{fname}\ntrue={true_str} | pred={pred_str} {correct} | '
        f'P(real)={prob_real:.3f} P(fake)={prob_fake:.3f}', fontsize=9)
    axes[0].imshow((img_np * 255).astype(np.uint8))
    axes[0].set_title('Original'); axes[0].axis('off')
    im = axes[1].imshow(mask_resized, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title('Attention Rollout'); axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay'); axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ── 평균 Attention Map 저장 ────────────────────────────────
def save_mean_attention(cls_masks_dict):
    cls_names = list(cls_masks_dict.keys())
    n_rows = len(cls_names)

    fig, axes = plt.subplots(n_rows, 1, figsize=(4, n_rows * 4))
    fig.suptitle('Mean Attention Rollout per Class (DLIB crop)', fontsize=13)

    for ax, cls_name in zip(axes, cls_names):
        masks = cls_masks_dict[cls_name]
        mean_mask = np.mean(masks, axis=0)
        mean_mask = (mean_mask - mean_mask.min()) / (mean_mask.max() - mean_mask.min() + 1e-8)
        mean_mask_resized = np.array(
            Image.fromarray((mean_mask * 255).astype(np.uint8)).resize((224, 224), Image.BICUBIC)
        ) / 255.0
        im = ax.imshow(mean_mask_resized, cmap='jet', vmin=0, vmax=1)
        ax.set_title(f'{cls_name} (n={len(masks)})', fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, 'mean_attention_map.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"평균 attention map 저장: {save_path}")

    npy_dir = os.path.join(SAVE_DIR, 'mean_masks_npy')
    os.makedirs(npy_dir, exist_ok=True)
    for cls_name, masks in cls_masks_dict.items():
        np.save(os.path.join(npy_dir, f'{cls_name}_mean.npy'), np.mean(masks, axis=0))
    print(f"mean masks npy 저장: {npy_dir}")

# ── 메인 ──────────────────────────────────────────────────
if __name__ == '__main__':
    random.seed(RANDOM_SEED)

    all_results    = []
    cls_masks_dict = {}
    loaded_ckpts   = {}
    rollouts       = {}

    for cls_name, (img_dir, true_label, max_n) in SAMPLES.items():
        print(f"\n=== {cls_name} ===")
        if not os.path.exists(img_dir):
            print(f"  Path not found: {img_dir}"); continue

        all_files = sorted([
            f for f in os.listdir(img_dir)
            if f.endswith('.png') or f.endswith('.jpg')
        ])
        if max_n and len(all_files) > max_n:
            all_files = random.sample(all_files, max_n)

        fas_ckpt = FAS_CKPTS[cls_name]
        if fas_ckpt not in loaded_ckpts:
            print(f"  Loading checkpoint: {fas_ckpt}")
            loaded_ckpts[fas_ckpt] = load_model(fas_ckpt).cuda()
            rollouts[fas_ckpt] = AttentionRollout(loaded_ckpts[fas_ckpt], discard_ratio=0.9)
        model  = loaded_ckpts[fas_ckpt]
        rollout = rollouts[fas_ckpt]

        os.makedirs(os.path.join(SAVE_DIR, cls_name), exist_ok=True)
        npy_dir = os.path.join(SAVE_DIR, cls_name, 'masks_npy')
        os.makedirs(npy_dir, exist_ok=True)

        preds      = []
        all_masks  = []
        n_skip     = 0

        for fname in all_files:
            img_path = os.path.join(img_dir, fname)
            img_tensor, img_np = preprocess(img_path)
            if img_tensor is None:
                n_skip += 1
                continue
            img_tensor = img_tensor.cuda()

            pred_class, prob_real, prob_fake = get_prediction(model, img_tensor)
            mask = rollout(img_tensor)
            np.save(os.path.join(npy_dir, fname.replace('.png', '.npy').replace('.jpg', '.npy')), mask)
            all_masks.append(mask)
            preds.append({
                'fname': fname, 'img_np': img_np, 'mask': mask,
                'pred_class': pred_class, 'prob_real': prob_real,
                'prob_fake': prob_fake, 'correct': pred_class == true_label,
            })

        cls_masks_dict[cls_name] = all_masks
        total = len(preds)
        print(f"  Skip(det fail)={n_skip} | Valid={total}")

        # 개별 저장 (최대 100장 랜덤)
        save_preds = random.sample(preds, min(MAX_SAVE, total))
        all_results.append({'class': cls_name, 'skipped': n_skip,
                             'total': total, 'saved': len(save_preds)})

        for i, p in enumerate(save_preds):
            save_path = os.path.join(SAVE_DIR, cls_name, f'{i:02d}.png')
            save_attention(p['img_np'], p['mask'], p['pred_class'],
                           p['prob_real'], p['prob_fake'],
                           true_label, save_path, p['fname'])
        print(f"  시각화: {len(save_preds)}장 | mask npy: {total}장 저장")

    print("\n=== 평균 Attention Map 저장 ===")
    save_mean_attention(cls_masks_dict)

    fig, ax = plt.subplots(figsize=(8, len(all_results) * 0.5 + 1.5))
    ax.axis('off')
    col_labels = ['class', 'skipped', 'total', 'saved']
    table_data = [[r[c] for c in col_labels] for r in all_results]
    table = ax.table(cellText=table_data, colLabels=col_labels, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(col_labels))))
    plt.title('FAS Model - Attention Rollout Statistics (DLIB crop)', fontsize=12, pad=10)
    plt.tight_layout()
    table_path = os.path.join(SAVE_DIR, 'statistics_table.png')
    plt.savefig(table_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 완료! 통계 표: {table_path}")