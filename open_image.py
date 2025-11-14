# open_image.py
import os
import cv2
import numpy as np
from typing import List

def open_images_in_path(path: str, gray=True, max_images=None) -> List[np.ndarray]:
    """
    Load images from folder. Returns list of HxW (grayscale) numpy arrays (uint8).
    Keeps original shapes where possible.
    """
    imgs = []
    for fn in sorted(os.listdir(path)):
        if max_images and len(imgs) >= max_images:
            break
        full = os.path.join(path, fn)
        if not os.path.isfile(full):
            continue
        # try to read
        if gray:
            img = cv2.imread(full, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(full, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[open_images_in_path] warning: could not read {full}")
            continue
        imgs.append(img)
    return imgs


# Robust image->angles extractor that ALWAYS returns exactly n_wires floats.
def image_to_angles(img, out_size=(8, 8), n_wires=4):
    """
    Convert a grayscale image (HxW or color) into n_wires angles (radians).
    - If input is color, it converts to grayscale.
    - Resizes to out_size, divides into 4 quadrants and computes mean brightness.
    - If n_wires != 4, resamples (linear interpolation) the 4 quadrant means to n_wires.
    Returns a numpy array dtype=float32 of shape (n_wires,).
    """
    import numpy as np
    from scipy.interpolate import interp1d

    # ensure grayscale 2D array
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)

    # normalize and resize
    small = cv2.resize(img, out_size, interpolation=cv2.INTER_AREA)
    h, w = small.shape
    # quadrants (always produces 4 values)
    patches = [(0, h//2, 0, w//2), (0, h//2, w//2, w),
               (h//2, h, 0, w//2), (h//2, h, w//2, w)]
    vals = []
    for (r0, r1, c0, c1) in patches:
        patch = small[r0:r1, c0:c1]
        if patch.size == 0:
            mean = 0.0
        else:
            mean = float(patch.mean()) / 255.0  # normalized to [0,1]
        vals.append((mean * 2.0 - 1.0) * (np.pi / 2.0))  # map to [-pi/2, pi/2]

    vals = np.array(vals, dtype=np.float32)  # shape (4,)

    # If requested wires != 4, resample via linear interpolation
    if n_wires != 4:
        xs = np.linspace(0.0, 1.0, num=4)
        f = interp1d(xs, vals, kind="linear", fill_value="extrapolate")
        new_xs = np.linspace(0.0, 1.0, num=n_wires)
        vals = f(new_xs).astype(np.float32)

    return vals  # shape (n_wires,)


# Utility: safe batch conversion from list of images -> 2D numpy array (N, n_wires)
def images_to_angles_batch(images, out_size=(8,8), n_wires=4):
    import numpy as np
    arrs = [image_to_angles(img, out_size=out_size, n_wires=n_wires) for img in images]
    return np.stack(arrs).astype(np.float32)
