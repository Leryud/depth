import cv2
import numpy as np
from typing import List
from src.profiling.profiler import profile


@profile
def resize_images(images: List[np.ndarray], scale: float) -> List[np.ndarray]:
    """
    Resize a list of images.

    Args:
        images (List[np.ndarray]): List of input images.
        scale (float): Scale factor for resizing.

    Returns:
        List[np.ndarray]: List of resized images.
    """
    return [cv2.resize(img, (0, 0), fx=scale, fy=scale) for img in images]


@profile
def combine_images(images: List[np.ndarray], horizontal: bool = True) -> np.ndarray:
    """
    Combine multiple images into a single image.

    Args:
        images (List[np.ndarray]): List of input images.
        horizontal (bool): If True, combine horizontally; otherwise, vertically.

    Returns:
        np.ndarray: Combined image.
    """
    if horizontal:
        return np.hstack(images)
    return np.vstack(images)


@profile
def add_labels(image: np.ndarray, labels: List[str]) -> np.ndarray:
    """
    Add labels to an image.

    Args:
        image (np.ndarray): Input image.
        labels (List[str]): List of labels to add.

    Returns:
        np.ndarray: Image with labels added.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = image.shape[:2]
    n = len(labels)
    for i, label in enumerate(labels):
        cv2.putText(
            image,
            label,
            (10 + i * (w // n), 30),
            font,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return image
