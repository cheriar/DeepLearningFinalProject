"""
Compute mCE (mean Corruption Error) on CIFAR-10-C

This script evaluates the model on all 15 corruption types to compute
the mean Corruption Error (mCE) metric as reported in the paper.

Expected result: mCE = 8.2 for ResNet18 with HA++ (PS)
"""
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from model.resnet import ResNet18
import numpy as np
from collections import OrderedDict
import os

# AlexNet error rates for normalization (from Hendrycks et al.)
# These are the baseline error rates used to normalize mCE
ALEXNET_ERR = {
    'gaussian_noise': 88.6,
    'shot_noise': 89.4,
    'impulse_noise': 89.2,
    'defocus_blur': 81.9,
    'glass_blur': 82.7,
    'motion_blur': 78.5,
    'zoom_blur': 79.8,
    'snow': 86.7,
    'frost': 82.7,
    'fog': 81.9,
    'brightness': 56.5,
    'contrast': 85.3,
    'elastic_transform': 84.7,
    'pixelate': 73.6,
    'jpeg_compression': 61.5,
}

CORRUPTION_TYPES = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]

def normalize(x):
    """Normalize images using CIFAR-10 mean/std"""
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    return (x - mean) / std

def load_model(checkpoint_path='checkpoints/resnet_cifar10_ha_p_.pth'):
    """Load pretrained ResNet18 model"""
    net = ResNet18(num_classes=10)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Remove 'module.' prefix from state dict keys
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
        
        # Convert to PIL Image
        from PIL import Image
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target
    
    def __len__(self):
        return len(self.data)

def evaluate_corruption(net, corruption_type, data_root='./data/CIFAR-10-C'):
    """Evaluate model on a specific corruption type across all 5 severity levels"""
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    print(f"  Loading {corruption_type} dataset...", end=" ", flush=True)
    # Load dataset for this corruption type
    dataset = CIFARC(root=data_root, corruption_type=corruption_type, transform=test_transform)
    print("Done!")
    
    # CIFAR-10-C has 50,000 images (10,000 per severity level)
    # We evaluate on all 5 severity levels
    total_images = len(dataset)
    images_per_severity = total_images // 5
    
    severity_errors = []
    
    for severity in range(5):
        print(f"    Severity {severity + 1}/5...", end=" ", flush=True)
        start_idx = severity * images_per_severity
        end_idx = (severity + 1) * images_per_severity
        
        # Create subset for this severity level
        subset = torch.utils.data.Subset(dataset, range(start_idx, end_idx))
        loader = DataLoader(subset, batch_size=128, shuffle=False, num_workers=0)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in loader:
                data = normalize(data)
                outputs = net(data, _eval=True)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        error = 100.0 * (1.0 - correct / total)
        severity_errors.append(error)
        print(f"Error: {error:.2f}%")
    
    # Return mean error across all 5 severity levels
    mean_error = np.mean(severity_errors)
    return mean_error, severity_errors

def compute_mCE(net, data_root='./data/CIFAR-10-C'):
    """Compute mean Corruption Error (mCE) across all corruption types"""
    
    print("\n" + "="*70)
    print(" Computing mCE on CIFAR-10-C")
    print(" This will evaluate 15 corruption types × 5 severity levels")
    print(" Estimated time: ~15-20 minutes on CPU")
    print("="*70)
    
    corruption_errors = {}
    normalized_errors = []
    
    for i, corruption in enumerate(CORRUPTION_TYPES, 1):
        print(f"\n[{i}/15] Evaluating: {corruption}")
        mean_error, severity_errors = evaluate_corruption(net, corruption, data_root)
        corruption_errors[corruption] = mean_error
        
        # Normalize by AlexNet error
        ce = 100.0 * mean_error / ALEXNET_ERR[corruption]
        normalized_errors.append(ce)
        
        print(f"  → Mean Error: {mean_error:.2f}% | CE: {ce:.2f}")
    
    # Compute mCE (mean of all normalized errors)
    mCE = np.mean(normalized_errors)
    
    print("\n" + "="*70)
    print(" RESULTS")
    print("="*70)
    print(f"\nmCE (mean Corruption Error): {mCE:.1f}")
    print(f"Expected from paper: 8.2")
    print(f"Difference: {abs(mCE - 8.2):.1f}")
    
    if abs(mCE - 8.2) < 0.5:
        print("\n✓ Successfully reproduced paper results!")
    else:
        print(f"\n⚠ Result differs from paper (tolerance: ±0.5)")
    
    print("\nPer-corruption breakdown:")
    print("-" * 70)
    for corruption in CORRUPTION_TYPES:
        ce = 100.0 * corruption_errors[corruption] / ALEXNET_ERR[corruption]
        print(f"  {corruption:20s}: Error={corruption_errors[corruption]:5.2f}%  CE={ce:5.2f}")
    print("="*70 + "\n")
    
    return mCE, corruption_errors

def main():
    print("\n" + "="*70)
    print(" CIFAR-10-C Corruption Robustness Evaluation")
    print(" Model: ResNet18 with HybridAugment++ (PS)")
    print("="*70)
    
    # Check if CIFAR-10-C exists
    data_root = './data/CIFAR-10-C'
    if not os.path.exists(data_root):
        print(f"\n✗ Error: CIFAR-10-C dataset not found at {data_root}")
        print("Please download and extract CIFAR-10-C.tar first.")
        return
    
    # Load model
    print("\nLoading pretrained ResNet18 model...")
    net = load_model()
    print("✓ Model loaded successfully")
    
    # Compute mCE
    mCE, corruption_errors = compute_mCE(net, data_root)

if __name__ == '__main__':
    main()
