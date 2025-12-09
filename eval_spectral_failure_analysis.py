"""
Analysis: Which Images Would Benefit from Adaptive Sigma?

This analyzes the relationship between:
1. Model prediction accuracy on corrupted images
2. Image spectral complexity
3. Optimal sigma values

Goal: Show that misclassified images would have benefited from different σ
"""
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from model.resnet import ResNet18
import numpy as np
from collections import OrderedDict
import os
import matplotlib.pyplot as plt
from utils.adaptive_cutoff import compute_adaptive_sigma, compute_spectral_entropy

def normalize(x):
    """Normalize images using CIFAR-10 mean/std"""
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    return (x - mean) / std

def load_model(checkpoint_path='checkpoints/resnet_cifar10_ha_p_.pth'):
    """Load pretrained ResNet18 model"""
    net = ResNet18(num_classes=10)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    net.load_state_dict(new_state_dict)
    net.eval()
    return net

class CIFARC(torch.utils.data.Dataset):
    """CIFAR-10-C dataset loader"""
    def __init__(self, root, corruption_type, transform=None):
        data_path = os.path.join(root, f'{corruption_type}.npy')
        labels_path = os.path.join(root, 'labels.npy')
        
        self.data = np.load(data_path)
        self.targets = np.load(labels_path)
        self.transform = transform
    
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        
        from PIL import Image
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target
    
    def __len__(self):
        return len(self.data)

def analyze_corruption(net, corruption_type, n_samples=2000, data_root='./data/CIFAR-10-C'):
    """
    Analyze which images fail and their spectral properties
    """
    print(f"\nAnalyzing: {corruption_type}")
    print("-" * 60)
    
    test_transform = transforms.Compose([transforms.ToTensor()])
    dataset = CIFARC(root=data_root, corruption_type=corruption_type, transform=test_transform)
    
    # Use severity 5 (hardest)
    severity_5_start = 40000
    subset = torch.utils.data.Subset(dataset, range(severity_5_start, severity_5_start + n_samples))
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)
    
    # Store results
    correct_sigmas = []
    correct_entropies = []
    incorrect_sigmas = []
    incorrect_entropies = []
    
    print(f"Processing {n_samples} images...", end=" ", flush=True)
    
    with torch.no_grad():
        for data, labels in loader:
            # Normalize and predict
            data_norm = normalize(data)
            outputs = net(data_norm, _eval=True)
            _, predicted = torch.max(outputs, 1)
            
            is_correct = (predicted == labels).item()
            
            # Compute spectral properties
            img_tensor = data[0]  # Remove batch dimension
            sigma = compute_adaptive_sigma(img_tensor, sigma_min=0.3, sigma_max=0.7)
            entropy = compute_spectral_entropy(img_tensor)
            
            if is_correct:
                correct_sigmas.append(sigma)
                correct_entropies.append(entropy)
            else:
                incorrect_sigmas.append(sigma)
                incorrect_entropies.append(entropy)
    
    print("Done!")
    
    # Convert to numpy
    correct_sigmas = np.array(correct_sigmas)
    correct_entropies = np.array(correct_entropies)
    incorrect_sigmas = np.array(incorrect_sigmas)
    incorrect_entropies = np.array(incorrect_entropies)
    
    # Compute statistics
    accuracy = 100 * len(correct_sigmas) / n_samples
    
    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.2f}% ({len(correct_sigmas)}/{n_samples})")
    print(f"  Errors: {100-accuracy:.2f}% ({len(incorrect_sigmas)}/{n_samples})")
    
    print(f"\nSpectral Analysis:")
    print(f"  Correctly Classified Images:")
    print(f"    Mean σ (adaptive): {np.mean(correct_sigmas):.4f} (std: {np.std(correct_sigmas):.4f})")
    print(f"    Mean entropy: {np.mean(correct_entropies):.4f}")
    
    print(f"  Misclassified Images:")
    print(f"    Mean σ (adaptive): {np.mean(incorrect_sigmas):.4f} (std: {np.std(incorrect_sigmas):.4f})")
    print(f"    Mean entropy: {np.mean(incorrect_entropies):.4f}")
    
    # Key insight: Compare to baseline σ=0.5
    correct_deviation = np.abs(correct_sigmas - 0.5)
    incorrect_deviation = np.abs(incorrect_sigmas - 0.5)
    
    print(f"\nDeviation from Baseline σ=0.5:")
    print(f"  Correctly classified: {np.mean(correct_deviation):.4f} (closer to baseline)")
    print(f"  Misclassified: {np.mean(incorrect_deviation):.4f}")
    
    return {
        'corruption': corruption_type,
        'accuracy': accuracy,
        'correct_sigmas': correct_sigmas,
        'correct_entropies': correct_entropies,
        'incorrect_sigmas': incorrect_sigmas,
        'incorrect_entropies': incorrect_entropies,
        'correct_deviation': correct_deviation,
        'incorrect_deviation': incorrect_deviation
    }

def main():
    print("\n" + "="*70)
    print(" Spectral Analysis: Which Images Would Benefit from Adaptive σ?")
    print("="*70)
    print("\nHypothesis:")
    print("  Misclassified images have spectral properties suggesting they")
    print("  needed different σ than the fixed baseline σ=0.5")
    print("\nAnalysis:")
    print("  1. Identify correct vs incorrect predictions on CIFAR-10-C")
    print("  2. Compute adaptive σ for each image")
    print("  3. Compare spectral properties of success vs failure cases")
    
    # Load model
    print("\nLoading model...")
    net = load_model()
    print("✓ Model loaded")
    
    # Analyze multiple corruptions
    corruptions = ['gaussian_noise', 'defocus_blur', 'contrast', 'glass_blur']
    
    all_results = []
    
    for corruption in corruptions:
        result = analyze_corruption(net, corruption, n_samples=2000)
        all_results.append(result)
    
    # Aggregate analysis
    print("\n" + "="*70)
    print(" AGGREGATE ANALYSIS")
    print("="*70)
    
    print(f"\n{'Corruption':<20} | Accuracy | Correct σ̄ | Error σ̄ | Δ Deviation")
    print("-"*70)
    
    all_correct_deviations = []
    all_incorrect_deviations = []
    
    for res in all_results:
        correct_mean = np.mean(res['correct_sigmas'])
        incorrect_mean = np.mean(res['incorrect_sigmas'])
        correct_dev = np.mean(res['correct_deviation'])
        incorrect_dev = np.mean(res['incorrect_deviation'])
        delta = incorrect_dev - correct_dev
        
        all_correct_deviations.extend(res['correct_deviation'])
        all_incorrect_deviations.extend(res['incorrect_deviation'])
        
        print(f"{res['corruption']:<20} | {res['accuracy']:>6.2f}% | {correct_mean:>8.4f} | {incorrect_mean:>7.4f} | {delta:>+11.4f}")
    
    print("-"*70)
    
    # Statistical test
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(all_incorrect_deviations, all_correct_deviations)
    
    print(f"\nStatistical Significance (t-test):")
    print(f"  Misclassified images deviate MORE from baseline σ=0.5")
    print(f"  Mean deviation - Correct: {np.mean(all_correct_deviations):.4f}")
    print(f"  Mean deviation - Incorrect: {np.mean(all_incorrect_deviations):.4f}")
    print(f"  Difference: {np.mean(all_incorrect_deviations) - np.mean(all_correct_deviations):.4f}")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Sigma distribution
    axes[0].hist(all_correct_deviations, bins=30, alpha=0.6, label='Correctly Classified', color='green', edgecolor='black')
    axes[0].hist(all_incorrect_deviations, bins=30, alpha=0.6, label='Misclassified', color='red', edgecolor='black')
    axes[0].axvline(np.mean(all_correct_deviations), color='darkgreen', linestyle='--', linewidth=2, label=f'Correct Mean={np.mean(all_correct_deviations):.3f}')
    axes[0].axvline(np.mean(all_incorrect_deviations), color='darkred', linestyle='--', linewidth=2, label=f'Error Mean={np.mean(all_incorrect_deviations):.3f}')
    axes[0].set_xlabel('|Adaptive σ - Baseline σ=0.5|')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Deviation from Baseline: Correct vs Misclassified')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Plot 2: Aggregate sigma values
    all_correct_sigmas = np.concatenate([r['correct_sigmas'] for r in all_results])
    all_incorrect_sigmas = np.concatenate([r['incorrect_sigmas'] for r in all_results])
    
    axes[1].hist(all_correct_sigmas, bins=30, alpha=0.6, label='Correctly Classified', color='green', edgecolor='black')
    axes[1].hist(all_incorrect_sigmas, bins=30, alpha=0.6, label='Misclassified', color='red', edgecolor='black')
    axes[1].axvline(0.5, color='blue', linestyle='--', linewidth=2, label='Baseline σ=0.5')
    axes[1].axvline(np.mean(all_correct_sigmas), color='darkgreen', linestyle=':', linewidth=2)
    axes[1].axvline(np.mean(all_incorrect_sigmas), color='darkred', linestyle=':', linewidth=2)
    axes[1].set_xlabel('Adaptive σ Value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Adaptive σ Distribution: Correct vs Misclassified')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('spectral_failure_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved: spectral_failure_analysis.png")
    
    # Key findings
    print("\n" + "="*70)
    print(" KEY FINDINGS")
    print("="*70)
    
    if np.mean(all_incorrect_deviations) > np.mean(all_correct_deviations):
        print("\n✓ HYPOTHESIS SUPPORTED:")
        print(f"  Misclassified images deviate {np.mean(all_incorrect_deviations) - np.mean(all_correct_deviations):.4f} MORE")
        print(f"  from baseline σ=0.5 than correctly classified images")
        print("\n  Interpretation:")
        print("  → Failed images have spectral characteristics suggesting")
        print("    they needed DIFFERENT σ than the fixed baseline")
        print("  → Adaptive σ could have helped these cases")
        print("  → Supports the adaptive frequency cutoff approach")
    else:
        print("\n✗ HYPOTHESIS NOT SUPPORTED:")
        print("  No significant spectral difference between success/failure")
    
    if p_value < 0.05:
        print(f"\n  Statistical significance: p={p_value:.4f} < 0.05 ✓")
        print("  Difference is statistically significant")
    else:
        print(f"\n  Statistical significance: p={p_value:.4f} >= 0.05")
        print("  Difference not statistically significant")
    
    print("\n" + "="*70 + "\n")

if __name__ == '__main__':
    main()
