
# data_utils/pseudo_mask.py
import numpy as np

def band_mask(h, w, band_ratio=0.12, sigma=2.0, thresh=0.25):
    '''
    Create a horizontal band-like pseudo mask centered vertically.
    Returns a float32 array in {0,1} of shape (h,w).
    '''
    cy = h / 2.0
    y = np.arange(h, dtype=np.float32)
    band = band_ratio * h
    g = np.exp(-0.5 * ((y - cy) / (sigma * band)) ** 2)  # vertical 1D Gaussian
    m = np.tile(g[:, None], (1, w))
    m = (m >= thresh).astype(np.float32)
    return m
