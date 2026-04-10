#!/usr/bin/env python3
"""
CelebA-Spoof 전처리 스크립트
txt 파일 기반으로 이미지 복사 및 224x224 리사이즈
"""
import os
from PIL import Image
from tqdm import tqdm

SRC_ROOT = '/media/NAS/DATASET/CelebA-Spoof/CelebA_Spoof_/CelebA_Spoof/Data'
DST_ROOT = '/media/NAS/DATASET/FAS_FSFM/celeb'
TXT_ROOT = '/home/junwoo/projects/FSFM/fsfm-3c/finuetune/cross_dataset_unseen_DiFF/data/MCIO/txt'
FRAME_SIZE = (224, 224)

def parse_filename(fname):
    """
    732_live_040170.jpg → (732, live, 040170.jpg)
    563_spoof_348234.jpg → (563, spoof, 348234.jpg)
    """
    parts = fname.split('_')
    id_ = parts[0]
    label = parts[1]  # live or spoof
    img_name = parts[2]
    return id_, label, img_name

def process_txt(txt_path, split, label):
    """real/fake txt 파일 처리"""
    dst_label = 'real' if label == 'real' else 'fake'
    dst_dir = os.path.join(DST_ROOT, split, dst_label)
    os.makedirs(dst_dir, exist_ok=True)

    lines = open(txt_path).readlines()
    success, fail = 0, 0

    for line in tqdm(lines, desc=f'{split}/{label}'):
        fname = line.strip().split('/')[-1]  # 732_live_040170.jpg
        id_, src_label, img_name = parse_filename(fname)

        # train/test 둘 다 찾아봄
        src_path = None
        for split_dir in ['train', 'test']:
            src_label_dir = 'live' if src_label == 'live' else 'spoof'
            candidate = os.path.join(SRC_ROOT, split_dir, id_, src_label_dir, img_name)
            if os.path.exists(candidate):
                src_path = candidate
                break

        if src_path is None:
            fail += 1
            continue

        dst_path = os.path.join(dst_dir, fname)
        try:
            img = Image.open(src_path).convert('RGB')
            img = img.resize(FRAME_SIZE, Image.BICUBIC)
            img.save(dst_path)
            success += 1
        except Exception as e:
            fail += 1

    print(f'{split}/{label}: success={success}, fail={fail}')

if __name__ == '__main__':
    process_txt(os.path.join(TXT_ROOT, 'celeb_real_train.txt'), 'train', 'real')
    process_txt(os.path.join(TXT_ROOT, 'celeb_fake_train.txt'), 'train', 'fake')
    print('완료!')
