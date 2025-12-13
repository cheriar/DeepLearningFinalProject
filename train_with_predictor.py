"""
Training with adaptive sigma predictor network

The predictor learns to map image spectral entropy -> optimal sigma
This is trained jointly with the classification model.
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
import numpy as np

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def compute_spectral_entropy(images):
    """
    Compute spectral entropy for a batch of images
    
    Args:
        images: [B, C, H, W] tensor
    Returns:
        entropy: [B] tensor of entropy values
    """
    batch_size = images.size(0)
    entropies = []
    
    for i in range(batch_size):
        img = images[i]  # [C, H, W]
        
        # Convert to grayscale for spectral analysis
        gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
        
        # FFT
        fft = torch.fft.fft2(gray)
        magnitude = torch.abs(fft)
        
        # Power spectrum
        power = magnitude ** 2
        power_flat = power.flatten()
        
        # Normalize to probability distribution
        power_sum = power_flat.sum()
        if power_sum > 0:
            prob = power_flat / power_sum
            # Shannon entropy
            prob = prob[prob > 1e-10]  # Avoid log(0)
            entropy = -(prob * torch.log2(prob)).sum()
        else:
            entropy = torch.tensor(0.0, device=images.device)
        
        entropies.append(entropy)
    
    return torch.stack(entropies)

class SigmaPredictor(nn.Module):
    """
    Neural network that predicts optimal sigma from spectral entropy
    
    Architecture: Simple MLP that learns the entropy -> sigma mapping
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
    def forward(self, entropy):
        """
        Args:
            entropy: [B] tensor of spectral entropy values
        Returns:
            sigma: [B] tensor of predicted sigma values in [0.1, 0.7]
        """
        # Normalize entropy to [0, 1] range (typical CIFAR entropy is ~8-12)
        entropy_norm = (entropy - 8.0) / 4.0
        entropy_norm = torch.clamp(entropy_norm, 0, 1).unsqueeze(1)  # [B, 1]
        
        # Predict sigma
        sigma_norm = self.network(entropy_norm).squeeze(1)  # [B]
        
        # Map [0, 1] -> [0.1, 0.7]
        sigma = 0.1 + 0.6 * sigma_norm
        
        return sigma

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

def hybrid_augment_with_predictor(x, sigma_predictor, ks=3, prob=0.6):
    """
    HybridAugment++ with learned adaptive sigma based on spectral entropy
    
    Args:
        x: input batch [B, C, H, W]
        sigma_predictor: SigmaPredictor network
        ks: kernel size
        prob: probability of applying augmentation
    """
    batch_size = x.size(0)
    
    # Always compute spectral entropy and sigma for stats tracking
    entropies = compute_spectral_entropy(x)
    sigmas = sigma_predictor(entropies)
    
    if random.uniform(0, 1) > prob:
        return x, sigmas, entropies
    
    index = torch.randperm(batch_size).to(x.device)
    
    # Apply Gaussian blur with per-image sigma
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
    
    return hybrid_ims, sigmas, entropies

def hybrid_augment_baseline(x, ks=3, sigma=0.5, prob=0.6):
    """Standard HybridAugment++ with fixed sigma"""
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

def train_epoch(model, sigma_predictor, train_loader, criterion, optimizer_model, optimizer_predictor, device, use_adaptive=False):
    """Train for one epoch"""
    model.train()
    if sigma_predictor is not None:
        sigma_predictor.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    sigma_stats = []
    entropy_stats = []
    
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        
        # Apply augmentation
        if use_adaptive:
            data_aug, sigmas, entropies = hybrid_augment_with_predictor(data, sigma_predictor, ks=3, prob=0.6)
            sigma_stats.extend(sigmas.detach().cpu().numpy())
            entropy_stats.extend(entropies.detach().cpu().numpy())
        else:
            data_aug = hybrid_augment_baseline(data, ks=3, sigma=0.5, prob=0.6)
        
        # Normalize
        data_norm = normalize(data)
        data_aug_norm = normalize(data_aug)
        
        # Combine original and augmented
        inputs = torch.cat([data_norm, data_aug_norm], 0)
        targets = torch.cat([labels, labels], 0)
        
        # Forward pass
        optimizer_model.zero_grad()
        if use_adaptive:
            optimizer_predictor.zero_grad()
        
        outputs = model(inputs, _eval=False)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer_model.step()
        if use_adaptive:
            optimizer_predictor.step()
        
        # Stats
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if (batch_idx + 1) % 100 == 0:
            if use_adaptive and len(sigma_stats) > 0:
                avg_sigma = np.mean(sigma_stats[-100:])
                avg_entropy = np.mean(entropy_stats[-100:])
                print(f'  Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}% | sigma_avg={avg_sigma:.3f} | H_avg={avg_entropy:.2f}')
            else:
                print(f'  Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%')
    
    avg_sigma = np.mean(sigma_stats) if len(sigma_stats) > 0 else None
    avg_entropy = np.mean(entropy_stats) if len(entropy_stats) > 0 else None
    
    return total_loss / len(train_loader), 100. * correct / total, avg_sigma, avg_entropy

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
    lr_model = 0.1
    lr_predictor = 0.001  # Lower LR for predictor
    
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
    
    # Create models directory
    os.makedirs('trained_models', exist_ok=True)
    
    criterion = nn.CrossEntropyLoss()
    
    # ========================================================================
    # Train Adaptive Model with Sigma Predictor
    # ========================================================================
    print("\n" + "="*70)
    print("TRAINING ADAPTIVE MODEL (Spectral Entropy -> Sigma Predictor)")
    print("="*70)
    
    model_adaptive = ResNet18(num_classes=10).to(device)
    sigma_predictor = SigmaPredictor(hidden_dim=64).to(device)
    
    # Separate optimizers
    optimizer_model = optim.SGD(
        model_adaptive.parameters(),
        lr=lr_model, momentum=0.9, weight_decay=5e-4
    )
    optimizer_predictor = optim.Adam(
        sigma_predictor.parameters(),
        lr=lr_predictor, weight_decay=1e-5
    )
    
    scheduler_model = optim.lr_scheduler.StepLR(optimizer_model, step_size=60, gamma=0.2)
    
    best_acc_adaptive = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Model LR: {scheduler_model.get_last_lr()[0]:.6f} | Predictor LR: {lr_predictor:.6f}")
        
        train_loss, train_acc, avg_sigma, avg_entropy = train_epoch(
            model_adaptive, sigma_predictor, train_loader, criterion,
            optimizer_model, optimizer_predictor, device, use_adaptive=True
        )
        test_loss, test_acc = test_epoch(model_adaptive, test_loader, criterion, device)
        
        print(f"Train - Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | sigma_avg={avg_sigma:.3f} | H_avg={avg_entropy:.2f}")
        print(f"Test  - Loss: {test_loss:.4f} | Acc: {test_acc:.2f}%")
        
        scheduler_model.step()
        
        # Save best model
        if test_acc > best_acc_adaptive:
            best_acc_adaptive = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_adaptive.state_dict(),
                'predictor_state_dict': sigma_predictor.state_dict(),
                'optimizer_model_state_dict': optimizer_model.state_dict(),
                'optimizer_predictor_state_dict': optimizer_predictor.state_dict(),
                'test_acc': test_acc,
            }, 'trained_models/resnet18_adaptive_predictor_best.pth')
            print(f"[OK] Saved best adaptive model (acc: {test_acc:.2f}%)")
    
    # Save final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model_adaptive.state_dict(),
        'predictor_state_dict': sigma_predictor.state_dict(),
        'test_acc': best_acc_adaptive,
    }, 'trained_models/resnet18_adaptive_predictor_final.pth')
    
    print(f"\n[OK] Adaptive training complete. Best accuracy: {best_acc_adaptive:.2f}%")
    
    # ========================================================================
    # Test predictor on sample entropies
    # ========================================================================
    print("\n" + "="*70)
    print("LEARNED ENTROPY -> SIGMA MAPPING")
    print("="*70)
    
    sigma_predictor.eval()
    test_entropies = torch.linspace(8.0, 12.0, 20).to(device)
    with torch.no_grad():
        predicted_sigmas = sigma_predictor(test_entropies).cpu().numpy()
    
    print("\nEntropy -> Predicted Sigma:")
    for entropy, sigma in zip(test_entropies.cpu().numpy(), predicted_sigmas):
        print(f"  H={entropy:.2f} -> sigma={sigma:.4f}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    
    print(f"\nAdaptive Model (Spectral Entropy Predictor):")
    print(f"  Best accuracy: {best_acc_adaptive:.2f}%")
    print(f"  Saved to: trained_models/resnet18_adaptive_predictor_best.pth")
    
    print(f"\nPredictor network learns:")
    print(f"  Input: Spectral entropy (frequency content measure)")
    print(f"  Output: Optimal blur sigma for that image")
    print(f"  Range: sigma in [0.1, 0.7]")

if __name__ == '__main__':
    main()
