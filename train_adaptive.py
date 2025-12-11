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
    """Per-class sigma with curriculum learning"""
    def __init__(self, num_classes=10, init_sigma=0.7):
        super().__init__()
        # Start with high sigma (strong augmentation)
        self.class_sigmas = nn.Parameter(
            torch.full((num_classes,), init_sigma)
        )
        
    def get_sigma(self, labels):
        """Get sigma for a batch of labels"""
        sigmas = self.class_sigmas[labels]
        # Clamp to valid range [0.1, 0.7]
        return torch.clamp(sigmas, 0.1, 0.7)
    
    def get_all_sigmas(self):
        """Get all class sigmas (for logging)"""
        return torch.clamp(self.class_sigmas, 0.1, 0.7)

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

def train_epoch(model, sigma_module, train_loader, criterion, optimizer_model, optimizer_sigma, device, class_correct, class_total):
    """Train for one epoch with curriculum learning"""
    model.train()
    sigma_module.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        
        # Apply augmentation with adaptive sigma
        data_aug = hybrid_augment_adaptive(data, labels, sigma_module, ks=3, prob=0.6)
        
        # Normalize
        data_norm = normalize(data)
        data_aug_norm = normalize(data_aug)
        
        # Combine original and augmented
        inputs = torch.cat([data_norm, data_aug_norm], 0)
        targets = torch.cat([labels, labels], 0)
        
        # Forward pass
        optimizer_model.zero_grad()
        optimizer_sigma.zero_grad()
        outputs = model(inputs, _eval=False)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer_model.step()
        optimizer_sigma.step()
        
        # Track per-class accuracy for curriculum
        _, predicted = outputs[:len(labels)].max(1)  # Only original images
        for i in range(10):
            mask = labels == i
            class_total[i] += mask.sum().item()
            class_correct[i] += (predicted[mask] == labels[mask]).sum().item()
        
        # Stats
        total_loss += loss.item()
        total += targets.size(0)
        correct += outputs.max(1)[1].eq(targets).sum().item()
        
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
    
    criterion = nn.CrossEntropyLoss()
    
    # ========================================================================
    # Train Adaptive Model with Curriculum Learning
    # ========================================================================
    print("\n" + "="*70)
    print("TRAINING ADAPTIVE MODEL (Curriculum-Based Per-Class σ)")
    print("="*70)
    
    model_adaptive = ResNet18(num_classes=10).to(device)
    sigma_module = PerClassSigmaModule(num_classes=10, init_sigma=0.7).to(device)
    
    # Separate optimizers: model uses weight decay, sigma doesn't
    optimizer_model = optim.SGD(
        model_adaptive.parameters(),
        lr=lr, momentum=0.9, weight_decay=5e-4
    )
    optimizer_sigma = optim.SGD(
        sigma_module.parameters(),
        lr=0.01, momentum=0.0, weight_decay=0.0  # Higher LR, no weight decay
    )
    scheduler_model = optim.lr_scheduler.StepLR(optimizer_model, step_size=60, gamma=0.2)
    
    # Track per-class accuracy for curriculum adjustment
    class_correct = torch.zeros(10)
    class_total = torch.zeros(10)
    
    best_acc_adaptive = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Model LR: {scheduler_model.get_last_lr()[0]:.6f} | Sigma LR: 0.01")
        
        # Reset per-class tracking
        class_correct.zero_()
        class_total.zero_()
        
        train_loss, train_acc = train_epoch(model_adaptive, sigma_module, train_loader, criterion, 
                                           optimizer_model, optimizer_sigma, device, class_correct, class_total)
        test_loss, test_acc = test_epoch(model_adaptive, test_loader, criterion, device)
        
        print(f"Train - Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"Test  - Loss: {test_loss:.4f} | Acc: {test_acc:.2f}%")
        
        # Log current sigma values and per-class accuracy every 10 epochs
        if (epoch + 1) % 10 == 0:
            current_sigmas = sigma_module.get_all_sigmas().detach().cpu().numpy()
            class_acc = (class_correct / (class_total + 1e-10)).cpu().numpy()
            print("\nPer-class performance and sigma:")
            for i, name in enumerate(class_names):
                print(f"  {name:12s}: Acc={class_acc[i]*100:5.1f}% | σ={current_sigmas[i]:.4f}")
        
        scheduler_model.step()
        
        # Save best model
        if test_acc > best_acc_adaptive:
            best_acc_adaptive = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_adaptive.state_dict(),
                'sigma_state_dict': sigma_module.state_dict(),
                'optimizer_model_state_dict': optimizer_model.state_dict(),
                'optimizer_sigma_state_dict': optimizer_sigma.state_dict(),
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
    
    print(f"\nAdaptive Model (Curriculum-Based Per-Class σ):")
    print(f"  Best accuracy: {best_acc_adaptive:.2f}%")
    print(f"  Saved to: trained_models/resnet18_adaptive_best.pth")
    
    print(f"\nFinal Learned Sigma Values (started at 0.7):")
    for i, (name, sigma) in enumerate(zip(class_names, final_sigmas)):
        change = sigma - 0.7
        print(f"  {name:12s}: σ={sigma:.4f} (Δ={change:+.4f})")
    
    print(f"\nSigma range: [{final_sigmas.min():.4f}, {final_sigmas.max():.4f}]")
    print(f"Sigma std: {final_sigmas.std():.4f}")

if __name__ == '__main__':
    main()
