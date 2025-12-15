"""
Safely load an adaptive checkpoint and save a cleaned, weights-only checkpoint
that contains only `model_state_dict` and `predictor_state_dict` (or `class_sigmas`).

Usage:
  python save_clean_checkpoint.py --in trained_models/resnet18_adaptive_best.pth \
                                  --out trained_models/resnet18_adaptive_clean.pth

This uses `torch.serialization.add_safe_globals` to allow numpy reconstruct when
loading checkpoints saved with older PyTorch/NumPy pickling behavior.
Only run this on checkpoints you trust (this repo's checkpoints are local and trusted).
"""
import argparse
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument('--in', dest='in_path', required=True)
parser.add_argument('--out', dest='out_path', required=True)
args = parser.parse_args()

in_path = args.in_path
out_path = args.out_path

if not os.path.exists(in_path):
    raise SystemExit(f"Input checkpoint not found: {in_path}")

print('Loading', in_path)
ckpt = None
try:
    ckpt = torch.load(in_path, map_location='cpu')
    print('Loaded checkpoint with keys:', list(ckpt.keys()))
except Exception as e:
    print('Primary load failed:', e)
    try:
        import numpy as _np
        from torch.serialization import add_safe_globals
        with add_safe_globals([_np._core.multiarray._reconstruct]):
            ckpt = torch.load(in_path, map_location='cpu', weights_only=False)
            print('Loaded with add_safe_globals; keys:', list(ckpt.keys()))
    except Exception as e2:
        print('Safe load failed:', e2)
        print('Trying final fallback (weights_only=False)')
        ckpt = torch.load(in_path, map_location='cpu', weights_only=False)
        print('Loaded fallback; keys:', list(ckpt.keys()))

if ckpt is None:
    raise SystemExit('Failed to load checkpoint')

# Build cleaned dict
clean = {'epoch': ckpt.get('epoch', None), 'test_acc': ckpt.get('test_acc', None)}
if 'model_state_dict' in ckpt:
    clean['model_state_dict'] = ckpt['model_state_dict']
if 'predictor_state_dict' in ckpt:
    clean['predictor_state_dict'] = ckpt['predictor_state_dict']
if 'class_sigmas' in ckpt:
    clean['class_sigmas'] = ckpt['class_sigmas']

print('Saving cleaned checkpoint to', out_path)
torch.save(clean, out_path)
print('Done.')
