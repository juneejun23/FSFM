# FSFM 논문 재현 실험

FSFM (A Generalizable Face Security Foundation Model via Self-Supervised Facial Representation Learning, CVPR 2025) 논문 재현 실험입니다.

원본 repo: https://github.com/wolo-wolo/FSFM-CVPR25

---

## 재현 결과 (Video-level AUC)

| 데이터셋 | 논문 수치 | 재현 결과 | 차이 |
|---|---|---|---|
| CDFv2 | 91.44 | 91.984 | +0.54 |
| DFDC | 83.47 | 83.990 | +0.52 |
| DFDCp | 89.71 | 90.509 | +0.80 |
| WDF | 86.96 | 86.208 | -0.75 |

---

## 환경 설정

### 1. Conda 가상환경 생성

```bash
conda create -n fsfm3c python=3.9.21
conda activate fsfm3c
```

### 2. PyTorch 설치 (CUDA 12.4 기준)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

CUDA 버전 확인:
```bash
nvidia-smi
```

### 3. 나머지 패키지 설치

requirements.txt에 torch 관련 줄이 있어서 제외하고 설치:
```bash
grep -v "torch" requirements.txt > requirements_notorch.txt
pip install -r requirements_notorch.txt
pip install huggingface_hub submitit tensorboard scikit-learn torchsummary timm==0.4.5
```

---

## Pretrained Checkpoint 다운로드

```bash
cd /home/<username>/projects/FSFM
python fsfm-3c/pretrain/download_pretrained_weitghts.py
```

다운로드된 checkpoint는 NAS로 이동:
```bash
mv ./checkpoint/pretrained_models /media/NAS/USERS/<username>/FSFM_checkpoints
ln -s /media/NAS/USERS/<username>/FSFM_checkpoints ./checkpoint/pretrained_models
```

또는 이미 다운로드된 checkpoint 사용:
```
/media/NAS/USERS/junwoo/FSFM_checkpoints/VF2_ViT-B/checkpoint-400.pth
```

---

## 데이터셋 다운로드

데이터셋은 NAS에 저장합니다.

### 1. 폴더 생성

```bash
mkdir -p /media/NAS/DATASET/FSFM/FaceForensics
mkdir -p /media/NAS/DATASET/FSFM/CelebDF-v2
mkdir -p /media/NAS/DATASET/FSFM/DFDC
mkdir -p /media/NAS/DATASET/FSFM/DFDCP
mkdir -p /media/NAS/DATASET/FSFM/WildDeepfake
```

### 2. HuggingFace에서 다운로드

```bash
conda activate jw  # tmux가 있는 환경
tmux new -s fsfm_download

conda activate fsfm3c
python -c "
from huggingface_hub import hf_hub_download

files = [
    ('finetune_datasets/deepfakes_detection/FaceForensics_FF++_all_cls.tar.gz', '/media/NAS/DATASET/FSFM/FaceForensics/'),
    ('finetune_datasets/deepfakes_detection/Celeb-DF-v2.tar.gz', '/media/NAS/DATASET/FSFM/CelebDF-v2/'),
    ('finetune_datasets/deepfakes_detection/DFDC.tar.gz', '/media/NAS/DATASET/FSFM/DFDC/'),
    ('finetune_datasets/deepfakes_detection/DFDCP.tar.gz', '/media/NAS/DATASET/FSFM/DFDCP/'),
    ('finetune_datasets/deepfakes_detection/deepfake_in_the_wild.tar.gz', '/media/NAS/DATASET/FSFM/WildDeepfake/'),
]

for filename, local_dir in files:
    print(f'Downloading {filename}...')
    hf_hub_download(
        repo_id='Wolowolo/DF_DiFF_FAS_dataset_in_FSFM_FSVFM',
        repo_type='dataset',
        filename=filename,
        local_dir=local_dir,
        resume_download=True
    )
    print(f'Done: {filename}')
"
```

세션 나오기: `Ctrl+B` → `D`

### 3. 압축 해제

```bash
cd /media/NAS/DATASET/FSFM

tar -xzvf FaceForensics/finetune_datasets/deepfakes_detection/FaceForensics_FF++_all_cls.tar.gz -C FaceForensics/ && \
tar -xzvf CelebDF-v2/finetune_datasets/deepfakes_detection/Celeb-DF-v2.tar.gz -C CelebDF-v2/ && \
tar -xzvf DFDC/finetune_datasets/deepfakes_detection/DFDC.tar.gz -C DFDC/ && \
tar -xzvf DFDCP/finetune_datasets/deepfakes_detection/DFDCP.tar.gz -C DFDCP/ && \
tar -xzvf WildDeepfake/finetune_datasets/deepfakes_detection/deepfake_in_the_wild.tar.gz -C WildDeepfake/ && \
echo "모든 압축 해제 완료"
```

---

## Fine-tuning (FF++로 학습)

```bash
cd /home/<username>/projects/FSFM/fsfm-3c/finuetune/cross_dataset_DfD

CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node=2 main_finetune_DfD.py \
    --batch_size 32 \
    --nb_classes 2 \
    --model vit_base_patch16 \
    --epochs 10 \
    --blr 2.5e-4 \
    --layer_decay 0.65 \
    --weight_decay 0.05 \
    --drop_path 0.1 \
    --reprob 0.25 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --dist_eval \
    --apply_simple_augment \
    --finetune '/media/NAS/USERS/junwoo/FSFM_checkpoints/VF2_ViT-B/checkpoint-400.pth' \
    --finetune_data_path '/media/NAS/DATASET/FSFM/FaceForensics/FaceForensics/32_frames/DS_FF++_all_cls/c23'
```

또는 이미 fine-tuning된 checkpoint 사용:
```
/media/NAS/USERS/junwoo/FSFM_checkpoints/finetuned_FF++/1266365/checkpoint-min_val_loss.pth
```

---

## Test (논문 수치 재현)

```bash
cd /home/<username>/projects/FSFM/fsfm-3c/finuetune/cross_dataset_DfD

CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 PYTHONWARNINGS="ignore" torchrun \
    --nproc_per_node=2 main_test_DfD.py \
    --eval \
    --apply_simple_augment \
    --model vit_base_patch16 \
    --nb_classes 2 \
    --batch_size 320 \
    --resume '/media/NAS/USERS/junwoo/FSFM_checkpoints/finetuned_FF++/1266365/checkpoint-min_val_loss.pth'
```

---

## 주의사항

### torch.load 수정 필요 (PyTorch 2.6 이상)

PyTorch 2.6부터 `torch.load` 기본값이 변경됨. 아래 파일들을 수정해야 함:

**main_finetune_DfD.py:**
```python
# 변경 전
checkpoint = torch.load(args.finetune, map_location='cpu')
# 변경 후
checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False)
```

**util/misc.py:**
```python
# 변경 전
checkpoint = torch.load(args.resume, map_location='cpu')
# 변경 후
checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
```

※ 이 repo는 이미 수정되어 있음.

---

## NAS 경로 정리

```
/media/NAS/DATASET/FSFM/
├── FaceForensics/     # FF++ (fine-tuning용)
├── CelebDF-v2/        # test
├── DFDC/              # test
├── DFDCP/             # test
└── WildDeepfake/      # test

/media/NAS/USERS/junwoo/FSFM_checkpoints/
├── VF2_ViT-B/                    # pretrained checkpoint
│   └── checkpoint-400.pth
└── finetuned_FF++/               # fine-tuned checkpoint
    └── 1266365/
        └── checkpoint-min_val_loss.pth
```