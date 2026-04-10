#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAS 데이터셋 변환 스크립트
- txt 파일 기반으로 정확한 train/test split 적용
- /media/NAS/DATASET/FAS_copy/face/ → /media/NAS/DATASET/FAS_FSFM/
- 256x256 → 224x224 리사이즈
- frame[6] 선택
- {video_prefix}_frame0.png 형식으로 저장
"""

import os
import re
from PIL import Image
from tqdm import tqdm

SRC_ROOT = '/media/NAS/DATASET/FAS_copy/face'
DST_ROOT = '/media/NAS/DATASET/FAS_FSFM'
TXT_ROOT = '/home/junwoo/projects/FSFM/fsfm-3c/finuetune/cross_dataset_unseen_DiFF/data/MCIO/txt'
FRAME_SIZE = (224, 224)
FRAME_IDX = 6  # 논문 기준 frame[6]

DATASET_SRC = {
    'casia': os.path.join(SRC_ROOT, 'CASIA-FASD'),
    'msu':   os.path.join(SRC_ROOT, 'MSU-MFSD', 'scene01'),
    'replay': os.path.join(SRC_ROOT, 'Idiap Replay-Attack'),
    'oulu':  os.path.join(SRC_ROOT, 'Oulu-NPU'),
}


def get_frame_path(video_dir):
    """비디오 폴더에서 frame[6] 경로 반환. 없으면 첫 번째 프레임."""
    if not os.path.exists(video_dir):
        return None
    frames = sorted([f for f in os.listdir(video_dir) if f.endswith('.png') or f.endswith('.jpg')])
    if not frames:
        return None
    idx = FRAME_IDX if FRAME_IDX < len(frames) else 0
    return os.path.join(video_dir, frames[idx])


def resize_and_save(src_path, dst_path):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    img = Image.open(src_path).convert('RGB')
    img = img.resize(FRAME_SIZE, Image.BICUBIC)
    img.save(dst_path)


def find_video_dir_casia(video_prefix):
    """
    video_prefix 예: 17_HR_1, 5_1, 3_NM_2
    CASIA 구조: train_release/{subject}/{video}/ 또는 test_release/{subject}/{video}/
    video_prefix = {subject}_{video}
    """
    # 마지막 _ 기준으로 subject와 video 분리
    # 예: 17_HR_1 → subject=17, video=HR_1
    #     5_1     → subject=5, video=1
    #     3_NM_2  → subject=3, video=NM_2
    parts = video_prefix.split('_')
    subject = parts[0]
    video = '_'.join(parts[1:])

    for split_dir in ['train_release', 'test_release']:
        path = os.path.join(DATASET_SRC['casia'], split_dir, subject, video)
        if os.path.exists(path):
            return path
    return None


def find_video_dir_msu(video_prefix):
    """
    video_prefix 예: real_client002_android_SD_scene01
    MSU 구조: scene01/real/{video}/ 또는 scene01/attack/{video}/
    """
    for label_dir in ['real', 'attack']:
        path = os.path.join(DATASET_SRC['msu'], label_dir, video_prefix)
        if os.path.exists(path):
            return path
    return None


def find_video_dir_replay(video_prefix):
    for split in ['train', 'test', 'devel']:
        # real: real_client... → real/client...
        if video_prefix.startswith('real_'):
            folder = video_prefix[len('real_'):]  # real_ 제거
            path = os.path.join(DATASET_SRC['replay'], split, 'real', folder)
            if os.path.exists(path):
                return path
        # fake: fixed_attack_... → attack/fixed/attack_...
        elif video_prefix.startswith('fixed_'):
            folder = video_prefix[len('fixed_'):]  # fixed_ 제거
            path = os.path.join(DATASET_SRC['replay'], split, 'attack', 'fixed', folder)
            if os.path.exists(path):
                return path
        elif video_prefix.startswith('hand_'):
            folder = video_prefix[len('hand_'):]  # hand_ 제거
            path = os.path.join(DATASET_SRC['replay'], split, 'attack', 'hand', folder)
            if os.path.exists(path):
                return path
    return None


def find_video_dir_oulu(video_prefix):
    """
    video_prefix 예: 3_1_07_1
    Oulu 구조: Train_files/{video}/ 또는 Test_files/{video}/ 또는 Dev_files/{video}/
    """
    for split_dir in ['Train_files', 'Test_files', 'Dev_files']:
        path = os.path.join(DATASET_SRC['oulu'], split_dir, video_prefix)
        if os.path.exists(path):
            return path
    return None


FIND_VIDEO_DIR = {
    'casia': find_video_dir_casia,
    'msu': find_video_dir_msu,
    'replay': find_video_dir_replay,
    'oulu': find_video_dir_oulu,
}


def convert_dataset(dataset_name):
    print(f"\n=== Converting {dataset_name} ===")

    # 해당 데이터셋의 모든 txt 파일 처리
    txt_files = [f for f in os.listdir(TXT_ROOT) if f.startswith(dataset_name + '_') and f.endswith('.txt')]
    txt_files = [f for f in txt_files if 'shot' not in f]  # shot 파일 제외

    all_entries = set()
    for txt_file in txt_files:
        with open(os.path.join(TXT_ROOT, txt_file)) as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        all_entries.update(lines)

    print(f"총 {len(all_entries)}개 항목 처리 중...")

    not_found = []
    for entry in tqdm(sorted(all_entries)):
        # entry 예: casia/train/real/17_HR_1_frame0.png
        parts = entry.split('/')
        # parts[0]=dataset, parts[1]=split, parts[2]=label, parts[3]=filename
        split = parts[1]   # train or test
        label = parts[2]   # real or fake
        filename = parts[3]  # 17_HR_1_frame0.png

        # video_prefix 추출: _frame0 제거
        video_prefix = filename.replace('_frame0.png', '').replace('_frame1.png', '')

        # 비디오 폴더 찾기
        video_dir = FIND_VIDEO_DIR[dataset_name](video_prefix)
        if video_dir is None:
            not_found.append(entry)
            continue

        # frame[6] 선택
        frame_path = get_frame_path(video_dir)
        if frame_path is None:
            not_found.append(entry)
            continue

        # 저장
        dst_path = os.path.join(DST_ROOT, dataset_name, split, label, filename)
        resize_and_save(frame_path, dst_path)

    print(f"완료. 못 찾은 항목: {len(not_found)}개")
    if not_found:
        print("못 찾은 항목 샘플:", not_found[:5])

if __name__ == '__main__':
    os.makedirs(DST_ROOT, exist_ok=True)
    convert_dataset('replay')  # replay만 재실행
    print("\n✅ 완료!")
    print(f"결과: {DST_ROOT}")