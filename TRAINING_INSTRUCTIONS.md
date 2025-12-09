# Training Instructions for GPU

## Setup
```bash
git clone https://github.com/cheriar/DeepLearningFinalProject.git
cd DeepLearningFinalProject/hybrid_augment
pip install torch torchvision numpy pillow matplotlib
```

## Download CIFAR-10
The script will auto-download CIFAR-10 to `./data/` on first run.

## Run Training
```bash
python train_adaptive.py
```

**Expected runtime:** 6-8 hours on RTX 3070 Ti

## What it does
1. Trains **baseline model** (fixed σ=0.5) for 200 epochs → saves to `trained_models/resnet18_baseline_best.pth`
2. Trains **adaptive model** (learnable per-class σ) for 200 epochs → saves to `trained_models/resnet18_adaptive_best.pth`
3. Logs learned sigma values every 10 epochs

## After Training
Run evaluation to compare models:
```bash
# Download CIFAR-10-C first (if not already downloaded)
python compare_models.py
```

This generates comprehensive comparison report and saves results to `comparison_results.npz`.

## Files to send back
- `trained_models/resnet18_baseline_best.pth`
- `trained_models/resnet18_adaptive_best.pth`
- `comparison_results.npz`
- Terminal output showing learned sigma values

---
**Questions?** Check that CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
