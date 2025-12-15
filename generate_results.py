"""
Generate figures and a results+discussion text file for the paper.
Saves plots to `outputs/` and writes `results_discussion.txt`.

Usage:
    python generate_results.py

The script is defensive: it will skip CIFAR-10-C evaluation if that dataset isn't present.
"""
import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision.transforms as T
from model.resnet import ResNet18

OUTDIR = 'outputs'
os.makedirs(OUTDIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Reuse SigmaPredictor definition (match training)
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


def compute_spectral_entropy_batch(images):
    # images: [B, C, H, W]
    B = images.size(0)
    device = images.device
    entropies = torch.zeros(B, device=device)
    for i in range(B):
        img = images[i]
        gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
        fft = torch.fft.fft2(gray)
        mag = torch.abs(fft)
        power = (mag ** 2).flatten()
        s = power.sum()
        if s > 0:
            prob = power / s
            prob = prob[prob > 1e-10]
            ent = -(prob * torch.log2(prob)).sum()
        else:
            ent = torch.tensor(0.0, device=device)
        entropies[i] = ent
    return entropies


def load_checkpoint(path, device=DEVICE):
    if not os.path.exists(path):
        return None
    try:
        # First try the default (weights_only may be True in newer PyTorch)
        ckpt = torch.load(path, map_location=device)
        return ckpt
    except Exception as e:
        print('Primary load failed for', path, '|', str(e))
        # Try a safer fallback: allow full unpickling with numpy global if available
        try:
            import numpy as _np
            from torch.serialization import add_safe_globals
            # Allow numpy reconstruct global during safe load; only do this for trusted checkpoints
            with add_safe_globals([_np._core.multiarray._reconstruct]):
                ckpt = torch.load(path, map_location=device, weights_only=False)
                return ckpt
        except Exception as e2:
            # Last resort: try loading with weights_only=False without extra globals (may be unsafe)
            try:
                ckpt = torch.load(path, map_location=device, weights_only=False)
                return ckpt
            except Exception as e3:
                print('Fallback loads failed for', path, '|', str(e2), str(e3))
                return None


def eval_clean(model, loader, device=DEVICE):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            # normalize
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1).to(device)
            std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1,3,1,1).to(device)
            data_norm = (data - mean) / std
            out = model(data_norm, _eval=True)
            pred = out.argmax(1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    return 100.0 * correct / total


def main():
    # Paths
    base_ckpt = 'trained_models/resnet18_baseline_best.pth'
    # try several possible adaptive checkpoint filenames (predictor vs class-sigma variants)
    adapt_ckpt_candidates = [
        'trained_models/resnet18_adaptive_best.pth',
    ]

    # Data
    transform = transforms.ToTensor()
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=4)

    # Load baseline
    base = load_checkpoint(base_ckpt)
    adapt = None
    adapt_path = None
    for p in adapt_ckpt_candidates:
        ck = load_checkpoint(p)
        if ck is not None:
            adapt = ck
            adapt_path = p
            break

    report_lines = []
    report_lines.append('Results and Discussion\n')
    report_lines.append('======================\n')

    if base is None:
        report_lines.append('Baseline checkpoint not found: %s\n' % base_ckpt)
    else:
        model_base = ResNet18(num_classes=10).to(DEVICE)
        model_base.load_state_dict(base['model_state_dict'])
        acc_base = eval_clean(model_base, test_loader)
        report_lines.append(f'Baseline clean accuracy: {acc_base:.2f}%')
        print('Baseline clean acc', acc_base)

    if adapt is None:
        report_lines.append('Adaptive predictor checkpoint not found (tried multiple paths)\n')
    else:
        model_adapt = ResNet18(num_classes=10).to(DEVICE)
        model_adapt.load_state_dict(adapt['model_state_dict'])
        predictor = SigmaPredictor(hidden_dim=64).to(DEVICE)
        # Determine what the checkpoint contains: predictor network or class_sigmas
        has_predictor = 'predictor_state_dict' in adapt
        has_class_sigmas = 'class_sigmas' in adapt
        if has_predictor:
            predictor.load_state_dict(adapt['predictor_state_dict'])
        elif has_class_sigmas:
            # create a trivial predictor that returns per-class sigma (wrap to tensor)
            class_sigmas = np.asarray(adapt['class_sigmas'])
            def class_based_predictor(entropy_tensor, labels=None):
                # labels must be provided when using class-based sigma; fallback returns mean
                if labels is None:
                    return torch.tensor(class_sigmas.mean(), device=entropy_tensor.device).repeat(entropy_tensor.size(0))
                else:
                    return torch.tensor(class_sigmas[labels.cpu().numpy()], device=entropy_tensor.device)
            predictor = class_based_predictor
        else:
            # no predictor info; set predictor to None to skip adaptive evaluation
            predictor = None

        # Evaluate clean accuracy
        acc_adapt = eval_clean(model_adapt, test_loader)
        report_lines.append(f'Adaptive (predictor) clean accuracy: {acc_adapt:.2f}%')
        print('Adaptive clean acc', acc_adapt)

        # Sample small subset to compute entropy->sigma mapping
        sample_loader = DataLoader(testset, batch_size=256, shuffle=True, num_workers=4)
        data_batch, _ = next(iter(sample_loader))
        data = data_batch.to(DEVICE)
        ent = compute_spectral_entropy_batch(data)
        if predictor is not None:
            with torch.no_grad():
                if callable(predictor):
                    try:
                        sig = predictor(ent)
                    except TypeError:
                        # class-based predictor expects labels; use mean
                        sig = predictor(ent)
                else:
                    sig = predictor(ent)
        else:
            sig = None
        ent_cpu = ent.cpu().numpy()
        sig_cpu = sig.cpu().numpy()

        # Plot entropy vs sigma if predictor produced sigmas
        if sig is not None:
            sig_cpu = sig.cpu().numpy()
            plt.figure(figsize=(6,4))
            plt.scatter(ent_cpu, sig_cpu, s=8, alpha=0.6)
            plt.xlabel('Spectral Entropy (bits)')
            plt.ylabel('Predicted σ')
            plt.title('Entropy → Predicted σ (sample)')
            plt.grid(True)
            fig1 = os.path.join(OUTDIR, 'entropy_vs_sigma.png')
            plt.savefig(fig1, dpi=200)
            plt.close()
            report_lines.append('\nSaved figure: ' + fig1)
        else:
            report_lines.append('\nSkipped entropy→sigma scatter: no predictor available in checkpoint')

        if sig is not None:
            # Histogram of predicted sigma
            plt.figure(figsize=(6,4))
            plt.hist(sig_cpu, bins=20, color='C0', alpha=0.8)
            plt.xlabel('Predicted σ')
            plt.ylabel('Count')
            plt.title('Distribution of predicted σ (sample)')
            fig2 = os.path.join(OUTDIR, 'sigma_hist.png')
            plt.savefig(fig2, dpi=200)
            plt.close()
            report_lines.append('Saved figure: ' + fig2)

        # Also show predictor mapping by sweeping entropy
        # Predictor curve (if available)
        ent_grid = torch.linspace(max(0.0, ent.min().item()-1.0), ent.max().item()+1.0, 100).to(DEVICE)
        if sig is not None:
            with torch.no_grad():
                sig_grid = predictor(ent_grid).cpu().numpy()
            plt.figure(figsize=(6,4))
            plt.plot(ent_grid.cpu().numpy(), sig_grid, '-C1')
            plt.xlabel('Spectral Entropy (bits)')
            plt.ylabel('Predicted σ')
            plt.title('Learned Entropy→σ mapping')
            fig3 = os.path.join(OUTDIR, 'entropy_sigma_curve.png')
            plt.savefig(fig3, dpi=200)
            plt.close()
            report_lines.append('Saved figure: ' + fig3)
        else:
            report_lines.append('Skipped entropy→σ curve: no predictor available in checkpoint')

    # CIFAR-10-C evaluation if available
    cifarc_dir = os.path.join('data', 'CIFAR-10-C')
    if os.path.isdir(cifarc_dir):
        corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                       'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                       'snow', 'frost', 'fog', 'brightness', 'contrast',
                       'elastic_transform', 'pixelate', 'jpeg_compression']
        severity = 5
        severity_start = (severity-1) * 10000
        baseline_scores = []
        adaptive_scores = []
        for corruption in corruptions:
            data_path = os.path.join(cifarc_dir, corruption + '.npy')
            labels_path = os.path.join(cifarc_dir, 'labels.npy')
            if not (os.path.exists(data_path) and os.path.exists(labels_path)):
                print('Missing CIFAR-10-C files; skipping further CIFAR-10-C eval')
                break
            imgs = np.load(data_path)
            labels = np.load(labels_path)
            imgs_sev = imgs[severity_start:severity_start+10000]
            labs_sev = labels[severity_start:severity_start+10000]
            # convert to tensor
            imgs_t = torch.tensor(imgs_sev.transpose(0,3,1,2)).float() / 255.0
            labs_t = torch.tensor(labs_sev).long()
            loader = DataLoader(torch.utils.data.TensorDataset(imgs_t, labs_t), batch_size=256, shuffle=False)
            # evaluate baseline
            if base is not None:
                correct_b = 0
                total = 0
                model_base.eval()
                with torch.no_grad():
                    for x, y in loader:
                        x, y = x.to(DEVICE), y.to(DEVICE)
                        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1).to(DEVICE)
                        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1,3,1,1).to(DEVICE)
                        x_norm = (x - mean) / std
                        out = model_base(x_norm, _eval=True)
                        pred = out.argmax(1)
                        correct_b += (pred == y).sum().item()
                        total += y.size(0)
                acc_b = 100.0 * correct_b / total
            else:
                acc_b = None
            # evaluate adaptive predictor
            if adapt is not None:
                correct_a = 0
                total = 0
                model_adapt.eval()
                predictor.eval()
                with torch.no_grad():
                    for x, y in loader:
                        x, y = x.to(DEVICE), y.to(DEVICE)
                        # compute entropies on CPU to match training use
                        ent = compute_spectral_entropy_batch(x)
                        sigs = predictor(ent)
                        # apply blur per-sample
                        lfc = torch.zeros_like(x)
                        for i in range(x.size(0)):
                            s = float(sigs[i].item())
                            b = T.GaussianBlur(kernel_size=3, sigma=s).to(DEVICE)
                            lfc[i] = b(x[i])
                        x_aug = lfc + (x - lfc)[torch.randperm(x.size(0))]
                        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1).to(DEVICE)
                        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1,3,1,1).to(DEVICE)
                        x_norm = (x_aug - mean) / std
                        out = model_adapt(x_norm, _eval=True)
                        pred = out.argmax(1)
                        correct_a += (pred == y).sum().item()
                        total += y.size(0)
                acc_a = 100.0 * correct_a / total
            else:
                acc_a = None

            baseline_scores.append(acc_b)
            adaptive_scores.append(acc_a)
            print(f'Corruption {corruption}: baseline={acc_b} adaptive={acc_a}')

        # Save corruption comparison figure if we collected scores
        if len(baseline_scores) == len(corruptions):
            x = np.arange(len(corruptions))
            b_scores = np.array([s if s is not None else np.nan for s in baseline_scores])
            a_scores = np.array([s if s is not None else np.nan for s in adaptive_scores])
            plt.figure(figsize=(10,4))
            plt.plot(x, b_scores, '-o', label='baseline')
            plt.plot(x, a_scores, '-o', label='adaptive')
            plt.xticks(x, corruptions, rotation=45, ha='right')
            plt.ylabel('Accuracy (%)')
            plt.title('CIFAR-10-C (severity 5) per-corruption accuracy')
            plt.legend()
            plt.tight_layout()
            fig4 = os.path.join(OUTDIR, 'cifar10c_comparison.png')
            plt.savefig(fig4, dpi=200)
            plt.close()
            report_lines.append('\nSaved figure: ' + fig4)

            # average
            avg_b = np.nanmean(b_scores)
            avg_a = np.nanmean(a_scores)
            report_lines.append(f'Average CIFAR-10-C (s=5) baseline: {avg_b:.2f}%')
            report_lines.append(f'Average CIFAR-10-C (s=5) adaptive: {avg_a:.2f}%')

    # Write report file
    report_path = os.path.join(OUTDIR, 'results_discussion.txt')
    with open(report_path, 'w') as f:
        for line in report_lines:
            f.write(line + '\n')

    print('\nWrote report to', report_path)

if __name__ == '__main__':
    main()
