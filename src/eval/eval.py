from typing import Dict

import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.dataset.nyu_dataloader import get_nyu_loader
from src.eval.metrics import RunningAverageDict, compute_metrics
from src.model.inference import process_frame
from src.model.loader import load_model
from src.utils.config import get_config


def normalize_depth(depth_map: np.ndarray) -> np.ndarray:
    """
    Normalize a depth map to the range [0, 1].

    Args:
        depth_map (np.ndarray): Input depth map.

    Returns:
        np.ndarray: Normalized depth map of the same shape.
    """
    min_depth = np.min(depth_map)
    max_depth = np.max(depth_map)
    return (depth_map - min_depth) / (max_depth - min_depth)


def evaluate(config: Dict) -> RunningAverageDict:
    """
    Evaluate the depth estimation model on the NYU dataset.

    Args:
        config (Dict): Configuration dictionary containing model and evaluation parameters.

    Returns:
        RunningAverageDict: Metrics collected during evaluation.
    """
    session = load_model(config["model"])
    metrics = RunningAverageDict()
    test_loader = get_nyu_loader(**config["eval"])

    for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
        image, depth = sample["image"], sample["depth"]
        image = np.array(image)

        depth_pred = process_frame(session, image, load=False)

        # Normalising to measure *relative* depth estimation
        depth = normalize_depth(depth.numpy())
        depth_pred = normalize_depth(depth_pred)

        # Inverting the depth scale to conform to the NYU dataset's format
        depth_pred = abs(1 - depth_pred)

        depth = torch.tensor(depth)
        depth_pred = torch.tensor(depth_pred)

        # Match ground truth resolution
        depth_pred = F.interpolate(
            depth_pred.unsqueeze(0),
            size=(config["eval"]["input_height"], config["eval"]["input_width"]),
            mode="bilinear",
            align_corners=True,
        ).squeeze(0)

        metrics.update(compute_metrics(depth, depth_pred[None], max_depth_eval=1))

    return metrics


def run_eval(eval_out: str):
    """
    Main function to run the evaluation and save results.
    """
    config = get_config("config/config_eval.yaml")
    metrics = evaluate(config)

    with open(eval_out, "w") as fp:
        json.dump(metrics.get_value(), fp, indent=4)

    return metrics
