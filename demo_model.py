import torch
import numpy as np
from torchvision import transforms, datasets
from model.resnet import ResNet18
from collections import OrderedDict
import matplotlib.pyplot as plt
import random
import os
from PIL import Image

def normalize(x):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    return (x - mean) / std

# class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

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

# Load model
model = ResNet18(num_classes=10)
checkpoint = torch.load('checkpoints/resnet_cifar10_ha_p_.pth', map_location='cpu')
new_state_dict = OrderedDict()
for k, v in checkpoint.items():
    name = k[7:] if k.startswith('module.') else k
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model.eval()

# Load datasets
test_transform = transforms.Compose([transforms.ToTensor()])
cifar10_clean = datasets.CIFAR10(root='./data', train=False, download=False, transform=test_transform)
cifar10_corrupted = CIFARC(root='./data/CIFAR-10-C', corruption_type='gaussian_noise', transform=test_transform)

# Select random images
clean_indices = random.sample(range(len(cifar10_clean)), 5)
corrupted_indices = random.sample(range(40000, 50000), 5)  # Severity 5

# Create visualization
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

# Top row: Clean images
with torch.no_grad():
    for idx, img_idx in enumerate(clean_indices):
        img, true_label = cifar10_clean[img_idx]
        
        # Predict
        img_norm = normalize(img.unsqueeze(0))
        output = model(img_norm, _eval=True)
        probs = torch.softmax(output, dim=1)
        pred_label = output.argmax(dim=1).item()
        confidence = probs.max().item()
        
        # Display
        ax = axes[0, idx]
        img_display = img.permute(1, 2, 0).numpy()
        ax.imshow(img_display)
        
        # Title with color based on correctness
        is_correct = (pred_label == true_label)
        color = 'green' if is_correct else 'red'
        
        title_text = f"True: {class_names[true_label]}\n"
        title_text += f"Pred: {class_names[pred_label]}\n"
        title_text += f"{confidence:.1%}"
        
        ax.set_title(title_text, fontsize=10, color=color, fontweight='bold')
        ax.axis('off')

# Bottom row: Corrupted images
with torch.no_grad():
    for idx, img_idx in enumerate(corrupted_indices):
        img, true_label = cifar10_corrupted[img_idx]
        
        # Predict
        img_norm = normalize(img.unsqueeze(0))
        output = model(img_norm, _eval=True)
        probs = torch.softmax(output, dim=1)
        pred_label = output.argmax(dim=1).item()
        confidence = probs.max().item()
        
        # Display
        ax = axes[1, idx]
        img_display = img.permute(1, 2, 0).numpy()
        ax.imshow(img_display)
        
        # Title with color based on correctness
        is_correct = (pred_label == true_label)
        color = 'green' if is_correct else 'red'
        
        title_text = f"True: {class_names[true_label]}\n"
        title_text += f"Pred: {class_names[pred_label]}\n"
        title_text += f"{confidence:.1%}"
        
        ax.set_title(title_text, fontsize=10, color=color, fontweight='bold')
        ax.axis('off')

# Add row labels
fig.text(0.02, 0.75, 'Clean\nImages', fontsize=12, fontweight='bold', 
         ha='center', va='center', rotation=0)
fig.text(0.02, 0.25, 'Corrupted\nImages\n(Gaussian\nNoise)', fontsize=12, 
         fontweight='bold', ha='center', va='center', rotation=0)

plt.suptitle('HybridAugment++ Model Demo: Clean vs Corrupted', 
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0.05, 0, 1, 0.96])
plt.savefig('model_demo.png', dpi=150, bbox_inches='tight')
plt.show()
