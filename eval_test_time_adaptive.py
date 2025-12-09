"""
Test-Time Adaptive Augmentation Evaluation

This tests whether adaptive sigma improves robustness even at test-time
without retraining the model. We apply augmentation during inference
and average predictions across multiple augmented views.
"""
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from model.resnet import ResNet18
import numpy as np
from collections import OrderedDict
import os
from datasets.APR import HybridAugmentPlusSingle

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

def evaluate_with_tta(net, corruption_type, adaptive=False, n_augmentations=5, data_root='./data/CIFAR-10-C'):
    """
    Evaluate with Test-Time Augmentation
    
    Args:
        net: Model
        corruption_type: Which corruption to test
        adaptive: Use adaptive sigma (True) or fixed sigma (False)
        n_augmentations: Number of augmented views per image
    """
    print(f"\n{'Adaptive' if adaptive else 'Fixed'} Augmentation (σ={'adaptive [0.3-0.7]' if adaptive else '0.5'})")
    print(f"  Corruption: {corruption_type}")
    print(f"  Augmentations per image: {n_augmentations}")
    
    # Create augmentation transform
    if adaptive:
        aug_transform = HybridAugmentPlusSingle(
            img_size=32, 
            ks=3, 
            sigma=0.5,  # This will be overridden by adaptive computation
            prob=1.0,   # Always apply augmentation
            adaptive_mode=True,
            sigma_min=0.3,
            sigma_max=0.7
        )
    else:
        aug_transform = HybridAugmentPlusSingle(
            img_size=32,
            ks=3,
            sigma=0.5,  # Fixed baseline sigma
            prob=1.0,
            adaptive_mode=False
        )
    
    # Load dataset - test on severity level 5 only (hardest)
    test_transform = transforms.Compose([transforms.ToTensor()])
    dataset = CIFARC(root=data_root, corruption_type=corruption_type, transform=test_transform)
    
    # Use only severity 5 (last 10,000 images) for speed
    severity_5_start = 40000  # 4 severities × 10,000
    severity_5_end = 50000
    subset = torch.utils.data.Subset(dataset, range(severity_5_start, severity_5_end))
    
    # Further subsample for speed (test on 1000 images instead of 10,000)
    subset = torch.utils.data.Subset(subset, range(0, 1000))
    
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)  # batch_size=1 for TTA
    
    correct = 0
    total = 0
    
    print("  Processing: ", end="", flush=True)
    
    with torch.no_grad():
        for idx, (data, labels) in enumerate(loader):
            if idx % 100 == 0:
                print(f"{idx}...", end="", flush=True)
            
            # Create multiple augmented views
            logits_sum = None
            
            for _ in range(n_augmentations):
                # Convert back to PIL for augmentation
                from torchvision.transforms import ToPILImage
                to_pil = ToPILImage()
                
                # Apply augmentation
                img_pil = to_pil(data[0])
                img_aug = aug_transform(img_pil)
                
                # Convert back to tensor and normalize
                img_tensor = transforms.ToTensor()(img_aug).unsqueeze(0)
                img_normalized = normalize(img_tensor)
                
                # Get prediction
                logits = net(img_normalized, _eval=True)
                
                if logits_sum is None:
                    logits_sum = logits
                else:
                    logits_sum += logits
            
            # Average predictions
            logits_avg = logits_sum / n_augmentations
            _, predicted = torch.max(logits_avg, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(" Done!")
    
    accuracy = 100 * correct / total
    error = 100 - accuracy
    
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Error: {error:.2f}%")
    
    return error

def main():
    print("\n" + "="*70)
    print(" Test-Time Adaptive Augmentation Evaluation")
    print("="*70)
    print("\nTesting whether adaptive sigma helps at inference time")
    print("(without retraining the model)")
    print("\nStrategy:")
    print("  - Apply augmentation during test-time")
    print("  - Average predictions across multiple augmented views")
    print("  - Compare fixed σ=0.5 vs adaptive σ∈[0.3, 0.7]")
    print("\nNote: Testing on subset (1000 images, severity 5) for speed")
    
    # Load model
    print("\nLoading model...")
    net = load_model()
    print("✓ Model loaded")
    
    # Test on a few representative corruptions
    test_corruptions = ['gaussian_noise', 'defocus_blur', 'contrast']
    n_augmentations = 5
    
    results = {}
    
    for corruption in test_corruptions:
        print("\n" + "-"*70)
        print(f"Testing: {corruption}")
        print("-"*70)
        
        # Baseline: Fixed sigma
        print("\n[1/2] Baseline (Fixed σ=0.5):")
        fixed_error = evaluate_with_tta(net, corruption, adaptive=False, n_augmentations=n_augmentations)
        
        # Adaptive: Adaptive sigma
        print("\n[2/2] Our Approach (Adaptive σ):")
        adaptive_error = evaluate_with_tta(net, corruption, adaptive=True, n_augmentations=n_augmentations)
        
        improvement = fixed_error - adaptive_error
        results[corruption] = {
            'fixed': fixed_error,
            'adaptive': adaptive_error,
            'improvement': improvement
        }
        
        print(f"\n  Summary for {corruption}:")
        print(f"    Fixed σ=0.5:     {fixed_error:.2f}% error")
        print(f"    Adaptive σ:      {adaptive_error:.2f}% error")
        print(f"    Improvement:     {improvement:+.2f}% {'✓' if improvement > 0 else '✗'}")
    
    # Final summary
    print("\n" + "="*70)
    print(" RESULTS SUMMARY")
    print("="*70)
    print(f"\nTest-Time Augmentation with {n_augmentations} views per image")
    print("Tested on 1000 images (severity 5) per corruption\n")
    
    print(f"{'Corruption':<20} | Fixed σ=0.5 | Adaptive σ | Improvement")
    print("-"*70)
    
    improvements = []
    for corruption, res in results.items():
        print(f"{corruption:<20} | {res['fixed']:>9.2f}% | {res['adaptive']:>9.2f}% | {res['improvement']:>+9.2f}%")
        improvements.append(res['improvement'])
    
    print("-"*70)
    avg_improvement = np.mean(improvements)
    print(f"{'Average':<20} | {'':>9} | {'':>9} | {avg_improvement:>+9.2f}%")
    
    print("\n" + "="*70)
    if avg_improvement > 0:
        print("✓ Adaptive sigma shows improvement at test-time!")
        print(f"  Average error reduction: {avg_improvement:.2f}%")
    else:
        print("⚠ No improvement observed (may need full retraining)")
    
    print("\nNote: This is test-time augmentation only.")
    print("Full validation requires retraining the model with adaptive σ.")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
