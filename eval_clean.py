"""Simple evaluation script for clean CIFAR-10 accuracy only"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model.resnet import ResNet18
import numpy as np

# Test transforms (normalization) - proper CIFAR-10 normalization
def normalize(x):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    return (x - mean) / std

def main():
    # Load model
    print("Loading ResNet18 model...")
    net = ResNet18(num_classes=10)

    # Load checkpoint
    checkpoint = torch.load('checkpoints/resnet_cifar10_ha_p_.pth', map_location='cpu')
    # Remove 'module.' prefix from state dict keys (DataParallel wrapper)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:] if k.startswith('module.') else k  # remove `module.`
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    net.eval()

    print("Model loaded successfully!")

    # Load CIFAR-10 test set (num_workers=0 to avoid Windows multiprocessing issues)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    testset = datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=test_transform)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)

    # Evaluate
    print("Evaluating on CIFAR-10 test set...")
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in testloader:
            # Normalize
            data = normalize(data)
            
            # Forward pass
            outputs = net(data, _eval=True)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"\n{'='*50}")
    print(f"CIFAR-10 Clean Test Accuracy: {accuracy:.2f}%")
    print(f"Correct: {correct}/{total}")
    print(f"{'='*50}\n")

if __name__ == '__main__':
    main()
