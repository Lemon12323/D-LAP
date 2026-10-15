"""Explicit lossless PNG serialization with a canonical integer epsilon guarantee.
Legacy utils.save_images is unchanged. Input tensors are never modified.
"""
from pathlib import Path
import numpy as np
from PIL import Image

def save_strict_image(img_tensor, canonical_clean_uint8, save_path, epsilon_pixels=16):
    if isinstance(epsilon_pixels, bool) or not isinstance(epsilon_pixels, int) or not 0 <= epsilon_pixels <= 255:
        raise ValueError('epsilon_pixels must be an integer between0 and255')
    clean = np.asarray(canonical_clean_uint8)
    if clean.dtype != np.uint8 or clean.ndim != 3 or clean.shape[2] != 3:
        raise ValueError('canonical_clean_uint8 must be an HWC RGB uint8 array')
    values = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    if values.shape != clean.shape or not np.isfinite(values).all():
        raise ValueError('Finite CHW RGB tensor must match canonical clean shape')
    # Explicit rule: clamp float values to[0,1], multiply in float32, truncate.
    candidate = (np.clip(values.astype(np.float32), 0., 1.) * np.float32(255.)).astype(np.uint8)
    anchor = clean.astype(np.int16)
    strict = np.clip(candidate.astype(np.int16), anchor - epsilon_pixels, anchor + epsilon_pixels)
    strict = np.clip(strict, 0, 255).astype(np.uint8)
    assert np.abs(strict.astype(np.int16)-anchor).max() <= epsilon_pixels
    Image.fromarray(strict, mode='RGB').save(Path(save_path), format='PNG')
