# FSFM 논문 재현 실험

FSFM (A Generalizable Face Security Foundation Model via Self-Supervised Facial Representation Learning, CVPR 2025) 논문 재현 실험입니다.

원본 repo: https://github.com/wolo-wolo/FSFM-CVPR25

---

## 재현 결과

### Table 1: Cross-dataset Deepfake Detection (Video-level AUC)

FF++ (c23)으로 학습 → unseen 데이터셋 테스트

| 데이터셋 | 논문 수치 | 재현 결과(2GPU) | 차이 |
|---|---|---|---|
| CDFv2 | 91.44 | **91.984** | +0.54 |
| DFDC | 83.47 | **83.990** | +0.52 |
| DFDCp | 89.71 | **90.509** | +0.80 |
| WDF | 86.96 | **86.208** | -0.75 |

✅ 논문 수치와 1% 이내로 일치. 재현 성공.

---

### Table 2: Cross-domain Face Anti-Spoofing (HTER% / AUC%)

MCIO 프로토콜 (0-shot): 4개 데이터셋 중 1개를 target으로, 나머지 + CelebA-Spoof로 학습

| 프로토콜 | 논문 HTER | 재현 HTER | 논문 AUC | 재현 AUC |
|---|---|---|---|---|
| ICM→O | - | 🔄 학습 중 | - | 🔄 학습 중 |
| OIM→C | - | 🔄 학습 중 | - | 🔄 학습 중 |
| OCM→I | - | 🔄 학습 중 | - | 🔄 학습 중 |
| OCI→M | - | 🔄 학습 중 | - | 🔄 학습 중 |

---

### Table 3: Unseen Diffusion Facial Forgery Detection (Frame-level AUC)

FF++ DeepFakes (c23)으로 학습 → DiFF benchmark 테스트

| Subset | 논문 수치 | 재현 결과(2 GPU) | 재현 결과(1 GPU) | 비고 |
|---|---|---|---|---|
| T2I | 61.74 | **82.549** | **87.104** | ※ |
| I2I | 71.91 | **80.218** | **84.049** | ※ |
| FS | 71.31 | **85.373** | **87.071** | ※ |
| FE | 78.98 | **79.081** | **84.703** | ✅ |



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

```bash
grep -v "torch" requirements.txt > requirements_notorch.txt
pip install -r requirements_notorch.txt
pip install huggingface_hub submitit tensorboard scikit-learn torchsummary timm==0.4.5
pip install pandas opencv-python-headless --break-system-packages
```

---

## PyTorch 2.6 호환성 수정 (이미 적용됨)

PyTorch 2.6부터 `torch.load` 기본값 변경으로 아래 파일들을 수정해야 함. 이 repo는 이미 수정되어 있음.

수정 내용: `torch.load(..., map_location='cpu')` → `torch.load(..., map_location='cpu', weights_only=False)`

수정 파일:
- `fsfm-3c/finuetune/cross_dataset_DfD/main_finetune_DfD.py`
- `fsfm-3c/finuetune/cross_dataset_unseen_DiFF/main_finetune_DiFF.py`
- `fsfm-3c/finuetune/cross_dataset_unseen_DiFF/main_test_DiFF.py`
- `fsfm-3c/util/misc.py`
- `fsfm-3c/finuetune/cross_domain_FAS/fas.py`

---

## NAS 현황

### Checkpoints `/media/NAS/USERS/junwoo/FSFM_checkpoints/`

```
VF2_ViT-B/
├── checkpoint-400.pth          # pretrained checkpoint (메인)
└── checkpoint-te-400.pth       # EMA checkpoint
```

> DfD fine-tuned checkpoint는 현재 로컬에만 있음:
> `/home/junwoo/projects/FSFM/fsfm-3c/finuetune/cross_dataset_DfD/checkpoint/.../checkpoint-min_val_loss.pth`
> DiFF fine-tuned checkpoint도 로컬에 있음:
> `./checkpoint/junwoo/experiments_finetune/2gpu_AUC_fixed/`

### 데이터셋 `/media/NAS/DATASET/`

**DfD / DiFF 데이터 (`FSFM/`)**
```
FSFM/
├── FaceForensics/   # FF++ 32_frames (fine-tuning용)
├── CelebDF-v2/      # DfD test
├── DFDC/            # DfD test
├── DFDCP/           # DfD test
├── WildDeepfake/    # DfD test
└── DiFF/            # DiFF test
```

**FAS 전처리 데이터 (`FAS_FSFM/`)** — face2 기반, 224×224, frame0/frame1
```
FAS_FSFM/
├── casia/  {train,test}/{real,fake}/
├── msu/    {train,test}/{real,fake}/
├── replay/ {train,test}/{real,fake}/
├── oulu/   {train,test}/{real,fake}/
└── celeb/  {real,fake}/             # CelebA-Spoof (각 40,000장)
```

**FAS 원본 데이터 (`FAS_copy/`)**
```
FAS_copy/
├── face/        # 256×256 리사이즈 버전
├── face2/       # MTCNN 원본 크롭 (전처리 소스)
├── face3/       # 256×256 리사이즈 버전
└── depthface2/  # depth map
```

**기타**
```
CelebA-Spoof/    # CelebA-Spoof 원본 (72GB)
celeba/          # CelebA (DDPM 학습용)
DFDP/            # deepfake video detection 데이터셋 (CLASS04)
cifar10/
mnist/
tiny_imagenet/
```

---

## Pretrained Checkpoint 다운로드

```bash
cd /home/<username>/projects/FSFM
python fsfm-3c/pretrain/download_pretrained_weitghts.py
```

또는 이미 다운로드된 checkpoint 사용:
```
/media/NAS/USERS/junwoo/FSFM_checkpoints/VF2_ViT-B/checkpoint-400.pth
```

---

## 데이터셋 다운로드

### 1. 폴더 생성

```bash
mkdir -p /media/NAS/DATASET/FSFM/FaceForensics
mkdir -p /media/NAS/DATASET/FSFM/CelebDF-v2
mkdir -p /media/NAS/DATASET/FSFM/DFDC
mkdir -p /media/NAS/DATASET/FSFM/DFDCP
mkdir -p /media/NAS/DATASET/FSFM/WildDeepfake
mkdir -p /media/NAS/DATASET/FSFM/DiFF
```

### 2. HuggingFace에서 다운로드 (tmux 권장)

```bash
python -c "
from huggingface_hub import hf_hub_download

files = [
    ('finetune_datasets/deepfakes_detection/FaceForensics_FF++_all_cls.tar.gz', '/media/NAS/DATASET/FSFM/FaceForensics/'),
    ('finetune_datasets/deepfakes_detection/FaceForensics_FF++_each_cls.tar.gz', '/media/NAS/DATASET/FSFM/FaceForensics/'),
    ('finetune_datasets/deepfakes_detection/Celeb-DF-v2.tar.gz', '/media/NAS/DATASET/FSFM/CelebDF-v2/'),
    ('finetune_datasets/deepfakes_detection/DFDC.tar.gz', '/media/NAS/DATASET/FSFM/DFDC/'),
    ('finetune_datasets/deepfakes_detection/DFDCP.tar.gz', '/media/NAS/DATASET/FSFM/DFDCP/'),
    ('finetune_datasets/deepfakes_detection/deepfake_in_the_wild.tar.gz', '/media/NAS/DATASET/FSFM/WildDeepfake/'),
    ('finetune_datasets/diffusion_facial_forgery_detection/DiFF.tar.gz', '/media/NAS/DATASET/FSFM/DiFF/'),
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

### 3. 압축 해제

```bash
cd /media/NAS/DATASET/FSFM

tar -xvf FaceForensics/finetune_datasets/deepfakes_detection/FaceForensics_FF++_all_cls.tar.gz -C FaceForensics/
tar -xvf FaceForensics/finetune_datasets/deepfakes_detection/FaceForensics_FF++_each_cls.tar.gz -C FaceForensics/
tar -xvf CelebDF-v2/finetune_datasets/deepfakes_detection/Celeb-DF-v2.tar.gz -C CelebDF-v2/
tar -xvf DFDC/finetune_datasets/deepfakes_detection/DFDC.tar.gz -C DFDC/
tar -xvf DFDCP/finetune_datasets/deepfakes_detection/DFDCP.tar.gz -C DFDCP/
tar -xvf WildDeepfake/finetune_datasets/deepfakes_detection/deepfake_in_the_wild.tar.gz -C WildDeepfake/
tar -xvf DiFF/finetune_datasets/diffusion_facial_forgery_detection/DiFF.tar.gz -C DiFF/
```

### 4. DiFF val 폴더 설정 (심볼릭 링크)

```bash
cd /media/NAS/DATASET/FSFM/DiFF/DiFF/test_subsets

ln -s /media/NAS/DATASET/FSFM/DiFF/DiFF/test_subsets/T2I/test T2I/val
ln -s /media/NAS/DATASET/FSFM/DiFF/DiFF/test_subsets/I2I/test I2I/val
ln -s /media/NAS/DATASET/FSFM/DiFF/DiFF/test_subsets/FS/test FS/val
ln -s /media/NAS/DATASET/FSFM/DiFF/DiFF/test_subsets/FE/test FE/val
```

### 5. FAS 데이터셋 준비

FAS 데이터는 별도 전처리가 필요합니다. `/media/NAS/DATASET/FAS_copy/face2/` 에 원본이 있어야 합니다.

```bash
# MCIO 데이터셋 전처리 (face2 기반, frame0/frame1 추출)
python /home/junwoo/projects/FSFM/datasets/finetune/preprocess/convert_v2.py

# CelebA-Spoof 전처리
python /home/junwoo/projects/FSFM/datasets/finetune/preprocess/convert_celeb.py
```

전처리 결과:
- CASIA: train(real=60, fake=180), test(real=90, fake=270) 비디오 × 2프레임
- MSU: train(real=30, fake=90), test(real=40, fake=120) 비디오 × 2프레임
- Replay: train(real=60, fake=300), test(real=80, fake=400) 비디오 × 2프레임
- Oulu: train(real=360, fake=1440), test(real=360, fake=1440) 비디오 × 2프레임
- CelebA-Spoof: train(real=40,000, fake=40,000) 이미지

---

## Table 1: DfD Fine-tuning 및 Test

### Fine-tuning (FF++로 학습)

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

### Test

```bash
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

## Table 2: FAS Fine-tuning 및 Test

### utils.py 경로 수정 (이미 적용됨)

```python
# fsfm-3c/finuetune/cross_domain_FAS/utils/utils.py
if dataset_name in ['casia', 'replay', 'oulu', 'msu']:
    root = '/media/NAS/DATASET/FAS_FSFM/'
    dataroot = '/home/junwoo/projects/FSFM/fsfm-3c/finuetune/cross_dataset_unseen_DiFF/data/MCIO/txt/'
elif dataset_name == 'celeb':
    root = '/media/NAS/DATASET/FAS_FSFM/'
    dataroot = '/home/junwoo/projects/FSFM/fsfm-3c/finuetune/cross_dataset_unseen_DiFF/data/MCIO/txt/'
```

### Fine-tuning (4개 프로토콜 × 5 runs)

```bash
cd /home/<username>/projects/FSFM/fsfm-3c/finuetune/cross_domain_FAS
bash run_all.sh
```

run_all.sh 내용:
```bash
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
```

- 각 config당 내부에서 seed 0~4로 5번 자동 반복 (총 4 × 5 = 20번 학습)
- config당 약 2시간, 전체 약 8시간 소요

---

## Table 3: DiFF Fine-tuning 및 Test

### Fine-tuning (FF++ DeepFakes subset으로 학습)

```bash
cd /home/<username>/projects/FSFM/fsfm-3c/finuetune/cross_dataset_unseen_DiFF

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node=1 main_finetune_DiFF.py \
    --batch_size 128 \
    --nb_classes 2 \
    --model vit_base_patch16 \
    --epochs 50 \
    --blr 5e-4 \
    --layer_decay 0.65 \
    --weight_decay 0.05 \
    --drop_path 0.1 \
    --reprob 0.25 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --normalize_from_IMN \
    --apply_simple_augment \
    --finetune '/media/NAS/USERS/junwoo/FSFM_checkpoints/VF2_ViT-B/checkpoint-400.pth' \
    --data_path '/media/NAS/DATASET/FSFM/FaceForensics/FaceForensics/32_frames/DS_FF++_each_cls/c23/DeepFakes' \
    --val_data_path '/media/NAS/DATASET/FSFM/DiFF/DiFF/test_subsets'
```

### Test

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node=1 main_test_DiFF.py \
    --normalize_from_IMN \
    --apply_simple_augment \
    --eval \
    --model vit_base_patch16 \
    --nb_classes 2 \
    --batch_size 320 \
    --resume '<path_to_finetuned_checkpoint_folder>'
```