from PIL import Image
import numpy as np
import math
from typing import Tuple

# ---- Utility ---------------------------------------------------------------
SAVE_PATH = "env/height_field.png"
try:
    RES_BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    RES_BICUBIC = Image.BICUBIC

def resize_bicubic(arr, size):
    h, w = size
    im = Image.fromarray((np.clip(arr,0,1)*255).astype(np.uint8))
    im = im.resize((w, h), resample=RES_BICUBIC)
    return np.asarray(im, dtype=np.float32)/255.0

def blur_down_up(arr, factor=2):
    h, w = arr.shape
    small = resize_bicubic(arr, (max(1,h//factor), max(1,w//factor)))
    return resize_bicubic(small, (h, w))

def gaussian_blur(arr, sigma_px=2.0):
    """Approximate Gaussian blur via repeated box blurs."""
    k = max(3, int(6*sigma_px+1)//2*2+1)  # odd kernel
    out = arr.copy()
    for _ in range(3):
        # horizontal
        c = np.cumsum(out, axis=1)
        left = np.pad(c[:, :-k], ((0,0),(k,0)), 'edge')
        right = c
        out = (right - left)/k
        # vertical
        c = np.cumsum(out, axis=0)
        left = np.pad(c[:-k, :], ((k,0),(0,0)), 'edge')
        right = c
        out = (right - left)/k
    return out

def normalize01(a):
    a = a - a.min()
    m = a.max()
    return a/m if m>1e-8 else np.zeros_like(a)

def save(a, path, gamma=1.0):
    a = np.clip(a,0,1)**gamma
    Image.fromarray((a*255).astype(np.uint8)).save(path)

# ---- Shape primitives ------------------------------------------------------
def add_dunes(img, amplitude, wavelength_px, direction_deg, asym=0.6):
    """Adds long, low dunes with asymmetric slip faces."""
    H, W = img.shape
    th = math.radians(direction_deg)
    Y, X = np.mgrid[0:H, 0:W]
    coord = X*math.cos(th) + Y*math.sin(th)
    waves = np.sin(2*np.pi*coord/max(1.0,wavelength_px))
    waves = np.where(waves>0, waves, asym*waves)
    img += amplitude * waves

def add_craters(img, count, rmin_px, rmax_px, dmin, dmax, seed=0):
    """Adds broad, shallow craters with gentle rims."""
    H, W = img.shape
    rng = np.random.default_rng(seed)
    Y, X = np.mgrid[0:H, 0:W]
    for _ in range(count):
        r_low = rmin_px
        r_high = min(rmax_px, 0.5*min(H, W) - 1)
        if r_high <= r_low:
            r_low, r_high = r_high/2, r_high  
        r = float(rng.uniform(r_low, r_high))
        depth = float(rng.uniform(dmin, dmax))
        cx = int(rng.integers(r, W-r))
        cy = int(rng.integers(r, H-r))
        d = np.sqrt((X-cx)**2 + (Y-cy)**2)
        sigma = 0.65*r
        bowl = np.exp(-(d**2)/(2*sigma**2))
        rim = np.exp(-((d - 0.9*r)**2)/(2*(0.28*r)**2))
        img -= depth*bowl
        img += 0.25*depth*rim

# add mares
def add_mares(img, count, rmin_px, rmax_px, dmin, dmax, seed=0):
    """Adds broad, shallow mares with gentle rims."""
    H, W = img.shape
    rng = np.random.default_rng(seed)
    Y, X = np.mgrid[0:H, 0:W]
    for _ in range(count):
        r_low = rmin_px
        r_high = min(rmax_px, 0.5*min(H, W) - 1)
        if r_high <= r_low:
            r_low, r_high = r_high/2, r_high  
        r = float(rng.uniform(r_low, r_high))
        depth = float(rng.uniform(dmin, dmax))
        cx = int(rng.integers(r, W-r))
        cy = int(rng.integers(r, H-r))
        d = np.sqrt((X-cx)**2 + (Y-cy)**2)
        sigma = 0.65*r
        bowl = np.exp(-(d**2)/(2*sigma**2))
        rim = np.exp(-((d - 0.9*r)**2)/(2*(0.28*r)**2))
        img -= depth*bowl
        img += 0.25*depth*rim

# add ridges
def add_ridges(img, count, length_px, width_px, height_px, seed=0):
    """Adds long, narrow ridges."""
    H, W = img.shape
    rng = np.random.default_rng(seed)
    Y, X = np.mgrid[0:H, 0:W]
    for _ in range(count):
        cx = int(rng.integers(0, W))
        cy = int(rng.integers(0, H))
        angle = rng.uniform(0, 2*math.pi)
        dx = math.cos(angle)
        dy = math.sin(angle)
        d = np.abs((X - cx)*dy - (Y - cy)*dx)
        ridge = np.exp(-(d**2)/(2*(width_px/2)**2))
        length_mask = np.maximum(0, 1 - (np.abs((X - cx)*dx + (Y - cy)*dy)/(length_px/2)))
        img += height_px * ridge * length_mask
# ---- Generate --------------------------------------------------------------
H = W = 1024  # image size
base = np.ones((H, W), dtype=np.float32)*0.5

# Gentle dunes (for 5m field, 1px≈5mm)
# add_dunes(base, amplitude=0.2, wavelength_px=900,  direction_deg=28, asym=0.6)
# add_dunes(base, amplitude=0.06, wavelength_px=1500, direction_deg=34, asym=0.6)

# # Sparse craters
# add_craters(base, count=20, rmin_px=100, rmax_px=240, dmin=0.015, dmax=0.08, seed=2025)

# # Sparse mares
add_mares(base, count=5, rmin_px=180, rmax_px=400, dmin=0.03, dmax=0.1, seed=2026)

# # Dense ridges
# add_ridges(base, count=30, length_px=700, width_px=18, height_px=0.07, seed=2027)
# Slightly more blur and reduced contrast to soften appearance
base = gaussian_blur(base, sigma_px=1.0)
base = np.clip(0.5 + 1.4*(base - 0.5), 0, 1)

save(base, SAVE_PATH, gamma=1.0)
print("Saved: height_field.png")
