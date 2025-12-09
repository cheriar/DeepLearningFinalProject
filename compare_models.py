"""
Compare baseline vs adaptive models on CIFAR-10-C

Evaluates both models and shows:
1. Overall performance (clean + corrupted)
2. Per-class performance breakdown
3. Learned sigma values vs fixed baseline
"""
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from model.resnet import ResNet18
import os
from PIL import Image

def normalize(x):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(x.device)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to(x.device)
    return (x - mean) / std

class CIFARC(torch.utils.data.Dataset):
    def __init__(self, root, corruption_type, transform=None):
        data_path = os.path.join(root, f'{corruption_type}.npy')
        labels_path = os.path.join(root, 'labels.npy')
        
        self.data = np.load(data_path)
        self.targets = np.load(labels_path)
        self.transform = transform
    
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    
    def __len__(self):
        return len(self.data)

def evaluate_model(model, loader, device):
    """Evaluate model and return per-class accuracy"""
    model.eval()
    
    class_correct = np.zeros(10)
    class_total = np.zeros(10)
    
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            data_norm = normalize(data)
            
            outputs = model(data_norm, _eval=True)
            _, predicted = outputs.max(1)
            
            for i in range(10):
                mask = labels == i
                class_total[i] += mask.sum().item()
                class_correct[i] += (predicted[mask] == labels[mask]).sum().item()
    
    class_acc = 100 * class_correct / (class_total + 1e-10)
    overall_acc = 100 * class_correct.sum() / class_total.sum()
    
    return overall_acc, class_acc, class_correct.sum(), class_total.sum()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Load models
    print("Loading models...")
    
    # Baseline model
    model_baseline = ResNet18(num_classes=10).to(device)
    checkpoint_baseline = torch.load('trained_models/resnet18_baseline_best.pth', map_location=device)
    model_baseline.load_state_dict(checkpoint_baseline['model_state_dict'])
    print(f"✓ Loaded baseline model (trained acc: {checkpoint_baseline['test_acc']:.2f}%)")
    
    # Adaptive model
    model_adaptive = ResNet18(num_classes=10).to(device)
    checkpoint_adaptive = torch.load('trained_models/resnet18_adaptive_best.pth', map_location=device)
    model_adaptive.load_state_dict(checkpoint_adaptive['model_state_dict'])
    learned_sigmas = checkpoint_adaptive['class_sigmas']
    print(f"✓ Loaded adaptive model (trained acc: {checkpoint_adaptive['test_acc']:.2f}%)")
    
    # Load test data
    print("\nLoading test datasets...")
    from torchvision import datasets
    test_transform = transforms.Compose([transforms.ToTensor()])
    
    cifar10_clean = datasets.CIFAR10(root='./data', train=False, download=False, transform=test_transform)
    test_loader_clean = DataLoader(cifar10_clean, batch_size=128, shuffle=False, num_workers=4)
    
    # ========================================================================
    # Clean CIFAR-10 Evaluation
    # ========================================================================
    print("\n" + "="*70)
    print("CLEAN CIFAR-10 EVALUATION")
    print("="*70)
    
    print("\nBaseline Model:")
    acc_baseline_clean, class_acc_baseline_clean, correct, total = evaluate_model(model_baseline, test_loader_clean, device)
    print(f"  Overall: {acc_baseline_clean:.2f}% ({int(correct)}/{int(total)})")
    
    print("\nAdaptive Model:")
    acc_adaptive_clean, class_acc_adaptive_clean, correct, total = evaluate_model(model_adaptive, test_loader_clean, device)
    print(f"  Overall: {acc_adaptive_clean:.2f}% ({int(correct)}/{int(total)})")
    
    print(f"\nImprovement: {acc_adaptive_clean - acc_baseline_clean:+.2f}%")
    
    # Per-class breakdown
    print("\n" + "-"*70)
    print("Per-Class Accuracy (Clean):")
    print("-"*70)
    print(f"{'Class':<12} | {'Baseline':<10} | {'Adaptive':<10} | {'Diff':<8} | {'Learned σ':<10}")
    print("-"*70)
    
    for i, name in enumerate(class_names):
        diff = class_acc_adaptive_clean[i] - class_acc_baseline_clean[i]
        print(f"{name:<12} | {class_acc_baseline_clean[i]:>6.2f}%   | {class_acc_adaptive_clean[i]:>6.2f}%   | {diff:>+5.2f}% | σ={learned_sigmas[i]:.4f}")
    
    # ========================================================================
    # CIFAR-10-C Evaluation
    # ========================================================================
    print("\n" + "="*70)
    print("CIFAR-10-C EVALUATION (Corrupted Images)")
    print("="*70)
    
    corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 
                   'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                   'snow', 'frost', 'fog', 'brightness',
                   'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    
    # Test on severity 5 (hardest)
    severity = 5
    severity_start = (severity - 1) * 10000
    
    results_baseline = {}
    results_adaptive = {}
    
    for corruption in corruptions:
        dataset = CIFARC(root='./data/CIFAR-10-C', corruption_type=corruption, transform=test_transform)
        
        # Get severity 5 subset
        indices = list(range(severity_start, severity_start + 10000))
        subset = torch.utils.data.Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=128, shuffle=False, num_workers=4)
        
        # Evaluate both models
        acc_baseline, class_acc_baseline, _, _ = evaluate_model(model_baseline, loader, device)
        acc_adaptive, class_acc_adaptive, _, _ = evaluate_model(model_adaptive, loader, device)
        
        results_baseline[corruption] = (acc_baseline, class_acc_baseline)
        results_adaptive[corruption] = (acc_adaptive, class_acc_adaptive)
        
        improvement = acc_adaptive - acc_baseline
        print(f"{corruption:<20} | Baseline: {acc_baseline:>5.2f}% | Adaptive: {acc_adaptive:>5.2f}% | Diff: {improvement:>+5.2f}%")
    
    # Average across corruptions
    avg_baseline = np.mean([results_baseline[c][0] for c in corruptions])
    avg_adaptive = np.mean([results_adaptive[c][0] for c in corruptions])
    avg_improvement = avg_adaptive - avg_baseline
    
    print("\n" + "-"*70)
    print(f"Average (all corruptions):")
    print(f"  Baseline: {avg_baseline:.2f}%")
    print(f"  Adaptive: {avg_adaptive:.2f}%")
    print(f"  Improvement: {avg_improvement:+.2f}%")
    
    # ========================================================================
    # Sigma Analysis
    # ========================================================================
    print("\n" + "="*70)
    print("LEARNED SIGMA ANALYSIS")
    print("="*70)
    
    baseline_sigma = 0.5
    print(f"\nBaseline (fixed): σ = {baseline_sigma:.4f} for all classes")
    print(f"\nAdaptive (learned per-class):")
    print(f"{'Class':<12} | {'Learned σ':<10} | {'Deviation from 0.5':<20}")
    print("-"*60)
    
    for i, name in enumerate(class_names):
        deviation = learned_sigmas[i] - baseline_sigma
        print(f"{name:<12} | {learned_sigmas[i]:.4f}     | {deviation:>+.4f}")
    
    print(f"\nSigma statistics:")
    print(f"  Mean:   {np.mean(learned_sigmas):.4f}")
    print(f"  Std:    {np.std(learned_sigmas):.4f}")
    print(f"  Min:    {np.min(learned_sigmas):.4f} ({class_names[np.argmin(learned_sigmas)]})")
    print(f"  Max:    {np.max(learned_sigmas):.4f} ({class_names[np.argmax(learned_sigmas)]})")
    print(f"  Range:  {np.max(learned_sigmas) - np.min(learned_sigmas):.4f}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\nClean CIFAR-10:")
    print(f"  Baseline: {acc_baseline_clean:.2f}%")
    print(f"  Adaptive: {acc_adaptive_clean:.2f}%")
    print(f"  → Improvement: {acc_adaptive_clean - acc_baseline_clean:+.2f}%")
    
    print(f"\nCIFAR-10-C (Average, Severity 5):")
    print(f"  Baseline: {avg_baseline:.2f}%")
    print(f"  Adaptive: {avg_adaptive:.2f}%")
    print(f"  → Improvement: {avg_improvement:+.2f}%")
    
    if avg_improvement > 0.5:
        print(f"\n✓ Adaptive approach shows meaningful improvement!")
    elif avg_improvement > 0:
        print(f"\n~ Adaptive approach shows modest improvement")
    else:
        print(f"\n✗ Adaptive approach did not improve performance")
    
    # Save results
    np.savez('comparison_results.npz',
             baseline_clean=acc_baseline_clean,
             adaptive_clean=acc_adaptive_clean,
             baseline_corrupted=avg_baseline,
             adaptive_corrupted=avg_adaptive,
             learned_sigmas=learned_sigmas,
             class_names=class_names)
    
    print(f"\n✓ Results saved to: comparison_results.npz")

if __name__ == '__main__':
    main()
