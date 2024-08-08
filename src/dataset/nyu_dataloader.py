import os
from typing import Dict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from src.dataset.preprocess import get_black_border, get_white_border
from src.depth_anything.util.transform import transform, transform_full_size

# The NYU dataloader has been adapted from https://github.com/LiheYoung/Depth-Anything/blob/main/metric_depth/zoedepth/data/data_mono.py


class NYUDataset(Dataset):
    """Dataset class for the NYU Depth V2 dataset."""

    def __init__(
        self,
        test_dir: str,
        input_res: str,
        input_height: int,
        input_width: int,
        min_depth: float,
        max_depth: float,
    ) -> None:
        """
        Initialize the NYUDataset.

        Args:
            test_dir (str): Directory containing the test images.
            input_res (str): Wether to use full resolution or reduced resolution as model input (518x518 vs 224x224).
            input_height (int): Height of the input images.
            input_width (int): Width of the input images.
            min_depth (float): Minimum depth value.
            max_depth (float): Maximum depth value.
        """
        self.test_dir = test_dir
        self.input_res = input_res
        self.image_files = sorted(
            [f for f in os.listdir(test_dir) if f.endswith("_colors.png")]
        )
        self.input_height = input_height
        self.input_width = input_width
        self.min_depth = min_depth
        self.max_depth = max_depth

    def __getitem__(self, idx: int):
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the item.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing the image and depth data.
        """
        img_name = self.image_files[idx]
        img_path = os.path.join(self.test_dir, img_name)
        depth_path = os.path.join(
            self.test_dir, img_name.replace("_colors.png", "_depth.png")
        )

        image = Image.open(img_path)
        depth_gt = Image.open(depth_path)
        w, h = image.size

        crop_params = get_white_border(np.array(image, dtype=np.uint8))
        crop_params_depth = get_black_border(np.array(depth_gt, dtype=np.uint8))

        image = image.crop(
            (crop_params.left, crop_params.top, crop_params.right, crop_params.bottom)
        )
        depth_gt = depth_gt.crop(
            (
                crop_params_depth.left,
                crop_params_depth.top,
                crop_params_depth.right,
                crop_params_depth.bottom,
            )
        )

        image = np.pad(
            np.array(image),
            (
                (crop_params.top, h - crop_params.bottom),
                (crop_params.left, w - crop_params.right),
                (0, 0),
            ),
            mode="reflect",
        )

        depth_gt = np.pad(
            np.array(depth_gt),
            (
                (crop_params_depth.top, h - crop_params_depth.bottom),
                (crop_params_depth.left, w - crop_params_depth.right),
            ),
            mode="constant",
            constant_values=0,
        )

        image = np.asarray(image, dtype=np.float32) / 255.0
        depth_gt = np.asarray(depth_gt, dtype=np.float32) / 1000.0

        if self.input_res == "full":
            sample = transform_full_size({"image": image, "depth": depth_gt})
        elif self.input_res == "reduced":
            sample = transform({"image": image, "depth": depth_gt})
        else:
            raise ValueError(
                "Working either with full or reduced input resolution. To change the resolution, check the transformations at src/depth_anything_v2/util/transform.py"
            )

        return {"image": sample["image"], "depth": sample["depth"]}

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Number of items in the dataset.
        """
        return len(self.image_files)


def get_nyu_loader(
    test_dir: str,
    input_res: str,
    input_height: int,
    input_width: int,
    min_depth: float,
    max_depth: float,
    batch_size: int,
) -> DataLoader:
    """
    Create a DataLoader for the NYU dataset.

    Args:
        test_dir (str): Directory containing the test images.
        input_height (int): Height of the input images.
        input_width (int): Width of the input images.
        min_depth (float): Minimum depth value.
        max_depth (float): Maximum depth value.
        batch_size (int): Batch size for the DataLoader.

    Returns:
        DataLoader: DataLoader for the NYU dataset.
    """
    dataset = NYUDataset(
        test_dir, input_res, input_height, input_width, min_depth, max_depth
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)


def denormalize(x: torch.Tensor) -> np.ndarray:
    """
    Reverse the ImageNet normalization applied to the input.

    Args:
        x (torch.Tensor): Input tensor of shape (N, 3, H, W).

    Returns:
        np.ndarray: Denormalized input as a NumPy array.
    """
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (torch.tensor(x) * std + mean).numpy()


def show_batch(batch: Dict[str, torch.Tensor], batch_size: int) -> None:
    """
    Display a batch of images and their corresponding depth maps.

    Args:
        batch (Dict[str, torch.Tensor]): Batch containing images and depth maps.
        batch_size (int): Number of samples in the batch.
    """
    fig, axes = plt.subplots(nrows=2, ncols=batch_size, figsize=(20, 5))

    for i in range(batch_size):
        img = denormalize(batch["image"][i])
        axes[0, i].imshow(img.squeeze().transpose(1, 2, 0))
        axes[0, i].set_title(f"Image {i+1}")
        axes[0, i].axis("off")

        axes[1, i].imshow(batch["depth"][i], cmap="gray")
        axes[1, i].set_title(f"Depth {i+1}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()


def test_dataloader(
    test_dir: str = "data/nyu2_test", batch_size: int = 8, samples_to_show: int = 1
) -> None:
    """
    Test the NYU dataset loader by displaying sample batches.

    Args:
        test_dir (str): Directory containing the test images.
        batch_size (int): Number of samples per batch.
        samples_to_show (int): Number of batches to display.
    """
    loader = get_nyu_loader(
        test_dir,
        input_res="reduced",
        input_height=480,
        input_width=640,
        min_depth=0.1,
        max_depth=10.0,
        batch_size=batch_size,
    )

    for i, samples in enumerate(loader):
        show_batch(samples, batch_size)
        if (i + 1) == samples_to_show:
            break


# Uncomment to test the dataloader
# test_dataloader()
