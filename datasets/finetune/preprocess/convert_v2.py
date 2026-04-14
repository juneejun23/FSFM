#!/usr/bin/env python3
"""
FAS 데이터셋 변환 스크립트 v2
- face2 (MTCNN 원본 크롭) 기반
- frame0 = frames[6], frame1 = frames[6 + len//2]
- 224x224 리사이즈
"""
import os
from PIL import Image
from tqdm import tqdm

SRC_ROOT = '/media/NAS/DATASET/FAS_copy/face2'
DST_ROOT = '/media/NAS/DATASET/FAS_FSFM'
TXT_ROOT = '/home/junwoo/projects/FSFM/fsfm-3c/finuetune/cross_dataset_unseen_DiFF/data/MCIO/txt'
FRAME_SIZE = (224, 224)

LABEL_DIR = {
    'real': 'live',
    'fake': 'spoof',
}

def get_frames(video_dir):
    """frame0 = frames[6], frame1 = frames[6 + len//2]"""
    if not os.path.exists(video_dir):
        return None, None
    frames = sorted([f for f in os.listdir(video_dir)
                     if f.endswith('.png') or f.endswith('.jpg')])
    if not frames:
        return None, None
    idx0 = 6 if 6 < len(frames) else 0
    idx1 = 6 + len(frames) // 2
    idx1 = idx1 if idx1 < len(frames) else len(frames) - 1
    return os.path.join(video_dir, frames[idx0]), os.path.join(video_dir, frames[idx1])

def save_img(src_path, dst_path):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    img = Image.open(src_path).convert('RGB')
    img = img.resize(FRAME_SIZE, Image.BICUBIC)
    img.save(dst_path)

def normalize_casia_prefix(video_prefix):
    """10_3 → 10_NM_3 변환 (숫자만 있는 경우 NM 타입)"""
    parts = video_prefix.split('_')
    # HR이나 NM이 이미 있으면 그대로
    if 'HR' in parts or 'NM' in parts:
        return video_prefix
    # 숫자_숫자 형식이면 NM 삽입
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        return f"{parts[0]}_NM_{parts[1]}"
    return video_prefix

def find_video_dir(dataset_name, video_prefix, label):
    label_dir = LABEL_DIR[label]
    base = os.path.join(SRC_ROOT, dataset_name)
    
    # casia는 NM 타입 매핑 시도
    prefixes = [video_prefix]
    if dataset_name == 'casia':
        prefixes.append(normalize_casia_prefix(video_prefix))
    
    for prefix in prefixes:
        for split in ['train', 'test', 'dev', 'devel',
                      'Train_files', 'Test_files', 'Dev_files',
                      'train_release', 'test_release']:
            path = os.path.join(base, split, label_dir, prefix)
            if os.path.exists(path):
                return path
    return None

def convert_dataset(dataset_name):
    print(f"\n=== Converting {dataset_name} (face2 기반) ===")

    txt_files = [f for f in os.listdir(TXT_ROOT)
                 if f.startswith(dataset_name + '_') and f.endswith('.txt')
                 and 'shot' not in f]

    all_entries = set()
    for txt_file in txt_files:
        with open(os.path.join(TXT_ROOT, txt_file)) as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        all_entries.update(lines)

    print(f"총 {len(all_entries)}개 항목 처리 중...")

    success, fail = 0, 0
    fail_list = []
    for entry in tqdm(sorted(all_entries)):
        parts = entry.split('/')
        split = parts[1]
        label = parts[2]
        filename = parts[3]

        video_prefix = filename.replace('_frame0.png', '').replace('_frame1.png', '')

        video_dir = find_video_dir(dataset_name, video_prefix, label)
        if video_dir is None:
            fail += 1
            fail_list.append(entry)
            continue

        frame0_path, frame1_path = get_frames(video_dir)
        if frame0_path is None:
            fail += 1
            fail_list.append(entry)
            continue

        dst0 = os.path.join(DST_ROOT, dataset_name, split, label,
                            video_prefix + '_frame0.png')
        dst1 = os.path.join(DST_ROOT, dataset_name, split, label,
                            video_prefix + '_frame1.png')
        save_img(frame0_path, dst0)
        save_img(frame1_path, dst1)
        success += 1

    print(f"완료: success={success}, fail={fail}")
    if fail_list:
        print("못 찾은 항목 샘플:", fail_list[:5])

if __name__ == '__main__':
    for ds in ['casia', 'msu', 'replay', 'oulu']:
        convert_dataset(ds)
    print("\n✅ 전체 완료!")
