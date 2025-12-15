"""
Demo: compare baseline model vs. adaptive predictor model

For a small random subset of CIFAR-10 test images this script:
- Loads baseline checkpoint at `trained_models/resnet18_baseline_best.pth`
- Loads adaptive model + predictor at `trained_models/resnet18_adaptive_predictor_best.pth`
- Computes spectral entropy for each image and the predictor's sigma
- Shows baseline prediction, adaptive model prediction on clean image,
  adaptive model prediction on blurred (predicted sigma) image, and sigma value

Run:
    python demo_compare_predictor.py --n 8

"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from model.resnet import ResNet18
import os

# Local copy of the SigmaPredictor and entropy utilities (matches training code)
class SigmaPredictor(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    def forward(self, entropy):
        entropy_norm = (entropy - 8.0) / 4.0
        entropy_norm = torch.clamp(entropy_norm, 0, 1).unsqueeze(1)
        sigma_norm = self.network(entropy_norm).squeeze(1)
        sigma = 0.1 + 0.6 * sigma_norm
        return sigma


def compute_spectral_entropy(images):
    # images: [B, C, H, W]
    entropies = []
    for i in range(images.size(0)):
        img = images[i]
        gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
        fft = torch.fft.fft2(gray)
        mag = torch.abs(fft)
        power = (mag ** 2).flatten()
        s = power.sum()
        if s > 0:
            prob = power / s
            prob = prob[prob > 1e-10]
            entropy = -(prob * torch.log2(prob)).sum()
        else:
            entropy = torch.tensor(0.0, device=images.device)
        entropies.append(entropy)
    return torch.stack(entropies)


def normalize(x):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1).to(x.device)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1,3,1,1).to(x.device)
    return (x - mean) / std

CLASS_NAMES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


def load_models(device, baseline_path='trained_models/resnet18_baseline_best.pth', adapt_path='trained_models/resnet18_adaptive_predictor_best.pth'):
    # Baseline
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Baseline checkpoint not found: {baseline_path}")
    baseline_ckpt = torch.load(baseline_path, map_location=device)
    model_base = ResNet18(num_classes=10).to(device)
    model_base.load_state_dict(baseline_ckpt['model_state_dict'])
    model_base.eval()

    # Adaptive (classifier + predictor)
    if not os.path.exists(adapt_path):
        raise FileNotFoundError(f"Adaptive checkpoint not found: {adapt_path}")
    adapt_ckpt = torch.load(adapt_path, map_location=device)
    model_adapt = ResNet18(num_classes=10).to(device)
    model_adapt.load_state_dict(adapt_ckpt['model_state_dict'])
    model_adapt.eval()
    predictor = SigmaPredictor(hidden_dim=64).to(device)
    predictor.load_state_dict(adapt_ckpt['predictor_state_dict'])
    predictor.eval()

    return model_base, model_adapt, predictor


def pil_from_tensor(tensor):
    # tensor in [C,H,W], range [0,1]
    img = tensor.permute(1,2,0).cpu().numpy()
    return img


def run_demo(n=8, device='cpu'):
    device = torch.device(device)
    transform = transforms.ToTensor()
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # pick random indices
    rng = np.random.RandomState(42)
    indices = rng.choice(len(testset), size=n, replace=False)
    subset = Subset(testset, indices)
    loader = DataLoader(subset, batch_size=n, shuffle=False)
    imgs, labels = next(iter(loader))  # get all at once

    model_base, model_adapt, predictor = load_models(device)

    imgs = imgs.to(device)
    labels = labels.to(device)

    # Baseline predictions on clean images
    with torch.no_grad():
        logits_base = model_base(normalize(imgs), _eval=True)
        probs_base = F.softmax(logits_base, dim=1)
        pred_base = probs_base.argmax(dim=1)

    # Predictor -> sigma
    entropies = compute_spectral_entropy(imgs)
    with torch.no_grad():
        sigmas = predictor(entropies.to(device))

    # Create blurred images (per-image sigma) to show what adaptive would apply
    blurrer = T.GaussianBlur(kernel_size=3, sigma=0.5)  # placeholder - we will apply per-sample below
    lfc = torch.zeros_like(imgs)
    for i in range(imgs.size(0)):
        s = float(sigmas[i].item())
        # use torchvision's functional via transform with a single image
        b = T.GaussianBlur(kernel_size=3, sigma=s)
        lfc[i] = b(imgs[i])
    hybrid_blur = lfc  # for visualization

    # Adaptive model predictions (clean and blurred)
    with torch.no_grad():
        logits_adapt_clean = model_adapt(normalize(imgs), _eval=True)
        probs_adapt_clean = F.softmax(logits_adapt_clean, dim=1)
        pred_adapt_clean = probs_adapt_clean.argmax(dim=1)

        logits_adapt_blur = model_adapt(normalize(hybrid_blur), _eval=True)
        probs_adapt_blur = F.softmax(logits_adapt_blur, dim=1)
        pred_adapt_blur = probs_adapt_blur.argmax(dim=1)

    # Print results and plot
    for i in range(imgs.size(0)):
        true = labels[i].item()
        print(f"Image {i+1}: True={CLASS_NAMES[true]}")
        print(f"  Baseline: {CLASS_NAMES[pred_base[i].item()]} (p={probs_base[i, pred_base[i]].item():.3f})")
        print(f"  Adaptive (clean): {CLASS_NAMES[pred_adapt_clean[i].item()]} (p={probs_adapt_clean[i, pred_adapt_clean[i]].item():.3f})")
        print(f"  Adaptive (blur, σ={sigmas[i].item():.3f}): {CLASS_NAMES[pred_adapt_blur[i].item()]} (p={probs_adapt_blur[i, pred_adapt_blur[i]].item():.3f})")
        print()

    # Show images in a grid
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(cols * 3, rows * 3))

    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        img = pil_from_tensor(imgs[i].cpu())
        plt.imshow(np.clip(img, 0, 1))
        plt.axis('off')
        title = f"T:{CLASS_NAMES[labels[i].item()]}\nB:{CLASS_NAMES[pred_base[i].item()]}\nA:{CLASS_NAMES[pred_adapt_blur[i].item()]}\nσ={sigmas[i].item():.3f}"
        plt.title(title, fontsize=9)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    run_demo(n=args.n, device=args.device)
