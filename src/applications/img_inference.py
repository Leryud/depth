import cv2
from src.model.loader import load_model
from src.model.inference import process_frame
from src.utils.image_processing import colorize_resize_depth_map
from src.utils.visualization import resize_images, combine_images, add_labels


def img_inference(config: dict, filepath: str):
    """
    Run inference on a single input image.

    Args:
        config (dict): Configuration dictionary.
        filepath (str): Path to the input image file.
    """

    # Load the model
    session = load_model(config["model"])

    # Load the image
    frame = cv2.imread(filepath)
    orig_shape = frame.shape[:2]

    # Check if the image was loaded successfully
    if frame is None:
        print(f"Error: Unable to load image at {filepath}")
        return

    depth_map = process_frame(session, frame)
    depth_color = colorize_resize_depth_map(depth_map, orig_shape)

    # Visualize results
    resized_images = resize_images([frame, depth_color], 0.5)
    combined_frame = combine_images(resized_images)
    labeled_frame = add_labels(combined_frame, ["Original", "Depth Map"])

    cv2.imshow("MiDaS Depth Estimation", labeled_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return depth_map
