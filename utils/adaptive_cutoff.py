"""
Adaptive Frequency Cutoff for HybridAugment++

This module implements adaptive sigma selection based on image spectral characteristics.
The key insight: simple images benefit from gentler augmentation (high sigma/cutoff),
while complex images can handle more aggressive frequency mixing (low sigma).
"""

import torch
import numpy as np
from scipy import stats


def compute_spectral_entropy(image_tensor):
    """
    Compute spectral entropy of an image as a measure of frequency complexity.
    
    Args:
        image_tensor: torch.Tensor of shape (C, H, W) or (H, W)
    
    Returns:
        float: Normalized spectral entropy in [0, 1]
    """
    # Convert to numpy and handle multi-channel
    if isinstance(image_tensor, torch.Tensor):
        img = image_tensor.cpu().numpy()
    else:
        img = image_tensor
    
    # If multi-channel, average across channels
    if img.ndim == 3:
        img = np.mean(img, axis=0)
    
    # Compute 2D FFT
    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)
    
    # Compute power spectrum
    power_spectrum = np.abs(fft_shift) ** 2
    
    # Normalize to probability distribution
    power_spectrum_norm = power_spectrum / (np.sum(power_spectrum) + 1e-10)
    
    # Compute entropy: -sum(p * log(p))
    entropy = stats.entropy(power_spectrum_norm.flatten() + 1e-10)
    
    # Normalize entropy to [0, 1] range
    # Theoretical max entropy for uniform distribution
    max_entropy = np.log(power_spectrum.size)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    return float(normalized_entropy)


def compute_adaptive_sigma(image_tensor, sigma_min=0.3, sigma_max=0.7, mode='entropy', alpha=3.0):
    """
    Compute adaptive sigma (frequency cutoff) based on image characteristics.
    
    Strategy:
    - High entropy (complex images with rich frequency content) → Lower sigma (more aggressive mixing)
    - Low entropy (simple images) → Higher sigma (gentler augmentation)
    
    Uses exponential mapping for information-theoretic justification:
    σ = σ_min + (σ_max - σ_min) × exp(-α × entropy)
    
    This preserves semantic content in simple images while allowing stronger augmentation
    for complex images that can tolerate more aggressive frequency perturbations.
    
    Args:
        image_tensor: torch.Tensor of shape (C, H, W)
        sigma_min: Minimum sigma value (most aggressive augmentation)
        sigma_max: Maximum sigma value (most gentle augmentation)
        mode: 'entropy' for spectral entropy, 'variance' for frequency variance
        alpha: Exponential decay rate (higher = more aggressive adaptation)
    
    Returns:
        float: Adaptive sigma value in [sigma_min, sigma_max]
    """
    if mode == 'entropy':
        complexity = compute_spectral_entropy(image_tensor)
    elif mode == 'variance':
        # Alternative: use frequency variance as complexity measure
        if isinstance(image_tensor, torch.Tensor):
            img = image_tensor.cpu().numpy()
        else:
            img = image_tensor
        
        if img.ndim == 3:
            img = np.mean(img, axis=0)
        
        fft = np.fft.fft2(img)
        fft_shift = np.fft.fftshift(fft)
        power_spectrum = np.abs(fft_shift) ** 2
        
        # Normalize variance to [0, 1]
        variance = np.var(power_spectrum)
        max_variance = np.max(power_spectrum) ** 2  # theoretical upper bound
        complexity = np.clip(variance / (max_variance + 1e-10), 0, 1)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Exponential mapping: high complexity → low sigma (aggressive augmentation)
    # σ = σ_min + (σ_max - σ_min) × exp(-α × entropy)
    # This provides stronger protection for low-entropy (fragile) images
    adaptive_sigma = sigma_min + (sigma_max - sigma_min) * np.exp(-alpha * complexity)
    
    return float(adaptive_sigma)


def compute_batch_adaptive_sigmas(image_batch, sigma_min=0.3, sigma_max=0.7, mode='entropy', alpha=3.0):
    """
    Compute adaptive sigma for a batch of images.
    
    Args:
        image_batch: torch.Tensor of shape (B, C, H, W)
        sigma_min: Minimum sigma value
        sigma_max: Maximum sigma value
        mode: Complexity measure to use
        alpha: Exponential decay rate
    
    Returns:
        list: List of adaptive sigma values, one per image
    """
    sigmas = []
    for i in range(image_batch.shape[0]):
        sigma = compute_adaptive_sigma(image_batch[i], sigma_min, sigma_max, mode, alpha)
        sigmas.append(sigma)
    
    return sigmas
