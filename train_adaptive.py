"""
Training script with per-class learnable sigma (Approach 2C)

This trains two models:
1. Baseline model (fixed sigma=0.5)
2. Adaptive model (learnable sigma per class)

Both models are saved for comparison.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms as T
from model.resnet import ResNet18
import random
from collections import OrderedDict
import numpy as np

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class PerClassSigmaModule(nn.Module):
    """Learnable sigma per class"""
    def __init__(self, num_classes=10, init_sigma=0.5):
        super().__init__()
        # Initialize all classes to baseline sigma
        self.class_sigmas = nn.Parameter(
            torch.full((num_classes,), init_sigma)
        )
        
    def get_sigma(self, labels):
        """Get sigma for a batch of labels"""
        sigmas = self.class_sigmas[labels]
        # Clamp to valid range [0.3, 0.7]
        return torch.clamp(sigmas, 0.3, 0.7)
    
    def get_all_sigmas(self):
        """Get all class sigmas (for logging)"""
        return torch.clamp(self.class_sigmas, 0.3, 0.7)

def normalize(x):
    """CIFAR-10 normalization"""
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(x.device)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to(x.device)
    return (x - mean) / std

def mix_data(x):
    """APR-Pair: amplitude-phase recombination"""
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    fft_1 = torch.fft.fftn(x, dim=(1,2,3))
    abs_1, angle_1 = torch.abs(fft_1), torch.angle(fft_1)
    
    fft_2 = torch.fft.fftn(x[index, :], dim=(1,2,3))
    abs_2, angle_2 = torch.abs(fft_2), torch.angle(fft_2)
    
    # Mix amplitudes with original phases
    fft_mixed = abs_2 * torch.exp(1j * angle_1)
    mixed_x = torch.fft.ifftn(fft_mixed, dim=(1,2,3)).real
    
    return mixed_x

def hybrid_augment_adaptive(x, labels, sigma_module, ks=3, prob=0.6):
    """
    HybridAugment++ with per-class adaptive sigma
    
    Args:
        x: input batch [B, C, H, W]
        labels: class labels [B]
        sigma_module: PerClassSigmaModule
        ks: kernel size
        prob: probability of applying augmentation
    """
    if random.uniform(0, 1) > prob:
        return x
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    # Get per-sample sigma based on class labels
    sigmas = sigma_module.get_sigma(labels)  # [B]
    
    # Apply Gaussian blur with per-sample sigma
    # Note: Need to apply blur individually since sigma varies per sample
    lfc = torch.zeros_like(x)
    for i in range(batch_size):
        sigma_val = sigmas[i].item()
        blurrer = T.GaussianBlur(kernel_size=ks, sigma=sigma_val)
        lfc[i] = blurrer(x[i])
    
    # High-frequency component
    hfc = x - lfc
    hfc_mix = hfc[index]
    
    # APR on low-frequency
    lfc_mixed = mix_data(lfc)
    
    # Combine
    hybrid_ims = lfc_mixed + hfc_mix
    
    return hybrid_ims

def hybrid_augment_baseline(x, ks=3, sigma=0.5, prob=0.6):
    """
    Standard HybridAugment++ with fixed sigma
    """
    if random.uniform(0, 1) > prob:
        return x
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    blurrer = T.GaussianBlur(kernel_size=ks, sigma=sigma)
    
    lfc = blurrer(x)
    hfc = x - lfc
    hfc_mix = hfc[index]
    
    # APR on low-frequency
    lfc_mixed = mix_data(lfc)
    
    # Combine
    hybrid_ims = lfc_mixed + hfc_mix
    
    return hybrid_ims

def train_epoch(model, sigma_module, train_loader, criterion, optimizer, device, use_adaptive=False):
    """Train for one epoch"""
    model.train()
    if sigma_module is not None:
        sigma_module.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        
        # Apply augmentation
        if use_adaptive:
            data_aug = hybrid_augment_adaptive(data, labels, sigma_module, ks=3, prob=0.6)
        else:
            data_aug = hybrid_augment_baseline(data, ks=3, sigma=0.5, prob=0.6)
        
        # Normalize
        data_norm = normalize(data)
        data_aug_norm = normalize(data_aug)
        
        # Combine original and augmented
        inputs = torch.cat([data_norm, data_aug_norm], 0)
        targets = torch.cat([labels, labels], 0)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs, _eval=False)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Stats
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if (batch_idx + 1) % 100 == 0:
            print(f'  Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%')
    
    return total_loss / len(train_loader), 100. * correct / total

def test_epoch(model, test_loader, criterion, device):
    """Test for one epoch"""
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            data_norm = normalize(data)
            
            outputs = model(data_norm, _eval=True)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(test_loader), 100. * correct / total

def main():
    # Settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    num_epochs = 200
    batch_size = 128
    lr = 0.1
    
    # Data
    print("\nLoading CIFAR-10...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Train set: {len(train_dataset)} images")
    print(f"Test set: {len(test_dataset)} images")
    
    # Class names for logging
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Create models directory
    os.makedirs('trained_models', exist_ok=True)
    
    # ========================================================================
    # Train Baseline Model (Fixed sigma=0.5)
    # ========================================================================
    print("\n" + "="*70)
    print("TRAINING BASELINE MODEL (Fixed σ=0.5)")
    print("="*70)
    
    model_baseline = ResNet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_baseline = optim.SGD(model_baseline.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler_baseline = optim.lr_scheduler.StepLR(optimizer_baseline, step_size=60, gamma=0.2)
    
    best_acc_baseline = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Learning rate: {scheduler_baseline.get_last_lr()[0]:.6f}")
        
        train_loss, train_acc = train_epoch(model_baseline, None, train_loader, criterion, optimizer_baseline, device, use_adaptive=False)
        test_loss, test_acc = test_epoch(model_baseline, test_loader, criterion, device)
        
        print(f"Train - Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"Test  - Loss: {test_loss:.4f} | Acc: {test_acc:.2f}%")
        
        scheduler_baseline.step()
        
        # Save best model
        if test_acc > best_acc_baseline:
            best_acc_baseline = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_baseline.state_dict(),
                'optimizer_state_dict': optimizer_baseline.state_dict(),
                'test_acc': test_acc,
            }, 'trained_models/resnet18_baseline_best.pth')
            print(f"✓ Saved best baseline model (acc: {test_acc:.2f}%)")
    
    # Save final baseline model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model_baseline.state_dict(),
        'test_acc': best_acc_baseline,
    }, 'trained_models/resnet18_baseline_final.pth')
    
    print(f"\n✓ Baseline training complete. Best accuracy: {best_acc_baseline:.2f}%")
    
    # ========================================================================
    # Train Adaptive Model (Per-class learnable sigma)
    # ========================================================================
    print("\n" + "="*70)
    print("TRAINING ADAPTIVE MODEL (Per-Class Learnable σ)")
    print("="*70)
    
    model_adaptive = ResNet18(num_classes=10).to(device)
    sigma_module = PerClassSigmaModule(num_classes=10, init_sigma=0.5).to(device)
    
    # Optimizer for both model and sigma parameters
    optimizer_adaptive = optim.SGD(
        list(model_adaptive.parameters()) + list(sigma_module.parameters()),
        lr=lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler_adaptive = optim.lr_scheduler.StepLR(optimizer_adaptive, step_size=60, gamma=0.2)
    
    best_acc_adaptive = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Learning rate: {scheduler_adaptive.get_last_lr()[0]:.6f}")
        
        # Log current sigma values every 10 epochs
        if epoch % 10 == 0:
            current_sigmas = sigma_module.get_all_sigmas().detach().cpu().numpy()
            print("\nCurrent per-class sigma values:")
            for i, (name, sigma) in enumerate(zip(class_names, current_sigmas)):
                print(f"  {name:12s}: σ={sigma:.4f}")
        
        train_loss, train_acc = train_epoch(model_adaptive, sigma_module, train_loader, criterion, optimizer_adaptive, device, use_adaptive=True)
        test_loss, test_acc = test_epoch(model_adaptive, test_loader, criterion, device)
        
        print(f"Train - Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"Test  - Loss: {test_loss:.4f} | Acc: {test_acc:.2f}%")
        
        scheduler_adaptive.step()
        
        # Save best model
        if test_acc > best_acc_adaptive:
            best_acc_adaptive = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_adaptive.state_dict(),
                'sigma_state_dict': sigma_module.state_dict(),
                'optimizer_state_dict': optimizer_adaptive.state_dict(),
                'test_acc': test_acc,
                'class_sigmas': sigma_module.get_all_sigmas().detach().cpu().numpy(),
            }, 'trained_models/resnet18_adaptive_best.pth')
            print(f"✓ Saved best adaptive model (acc: {test_acc:.2f}%)")
    
    # Save final adaptive model
    final_sigmas = sigma_module.get_all_sigmas().detach().cpu().numpy()
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model_adaptive.state_dict(),
        'sigma_state_dict': sigma_module.state_dict(),
        'test_acc': best_acc_adaptive,
        'class_sigmas': final_sigmas,
    }, 'trained_models/resnet18_adaptive_final.pth')
    
    print(f"\n✓ Adaptive training complete. Best accuracy: {best_acc_adaptive:.2f}%")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\nBaseline Model (Fixed σ=0.5):")
    print(f"  Best accuracy: {best_acc_baseline:.2f}%")
    print(f"  Saved to: trained_models/resnet18_baseline_best.pth")
    
    print(f"\nAdaptive Model (Per-Class Learnable σ):")
    print(f"  Best accuracy: {best_acc_adaptive:.2f}%")
    print(f"  Saved to: trained_models/resnet18_adaptive_best.pth")
    
    print(f"\nLearned Sigma Values:")
    for i, (name, sigma) in enumerate(zip(class_names, final_sigmas)):
        print(f"  {name:12s}: σ={sigma:.4f}")
    
    print(f"\nImprovement: {best_acc_adaptive - best_acc_baseline:+.2f}%")

if __name__ == '__main__':
    main()
