import cv2
import numpy as np
from src.profiling.profiler import profile


@profile
def colorize_resize_depth_map(
    depth_map: np.ndarray,
    orig_shape: tuple[int, ...],
    colormap: int = cv2.COLORMAP_INFERNO,
) -> np.ndarray:
    """
    Colorize a depth map.

    Args:
        depth_map (np.ndarray): Input depth map.
        colormap (int): OpenCV colormap to use.

    Returns:
        np.ndarray: Colorized depth map.
    """
    orig_h, orig_w = orig_shape
    depth_map = depth_map.transpose(1, 2, 0)
    depth_map = cv2.resize(depth_map, (orig_w, orig_h))
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)

    depth_map = depth_map.astype(np.uint8)

    return cv2.applyColorMap(depth_map, colormap)


@profile
def overlay_images(
    img1: np.ndarray, img2: np.ndarray, alpha: float = 0.6
) -> np.ndarray:
    """
    Overlay two images with transparency.

    Args:
        img1 (np.ndarray): First image.
        img2 (np.ndarray): Second image.
        alpha (float): Transparency factor.

    Returns:
        np.ndarray: Overlayed image.
    """
    return cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
