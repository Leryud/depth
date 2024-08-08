import cv2
import time
from src.model.loader import load_model
from src.model.inference import process_frame
from src.utils.image_processing import colorize_resize_depth_map
from src.utils.visualization import combine_images, add_labels
from src.profiling.profiler import profiler


def fmv_inference(config: dict, filepath: str):
    """
    Run real-time inference on a input video.

    Args:
        config (dict): Configuration dictionary.
    """

    session = load_model(config["model"])

    cap = cv2.VideoCapture(filepath)

    profiler.start_video_profile("fmv_inference")

    while cap.isOpened():
        ret, frame = cap.read()
        frame_time_input = time.time()
        if not ret:
            break

        orig_shape = frame.shape[:2]
        depth_map = process_frame(session, frame)
        # depth_map = np.asarray(depth_map, dtype=np.float32) # For fp16 inference
        depth_color = colorize_resize_depth_map(depth_map, orig_shape)

        frame_time_output = time.time()
        latency = frame_time_output - frame_time_input
        profiler.profile_frame("fmv_inference", latency)

        combined_frame = combine_images([frame, depth_color])
        labeled_frame = add_labels(
            combined_frame, ["Original", f"Depth Map | Latency: {latency:.4f}"]
        )

        cv2.imshow("MiDaS Depth Estimation", labeled_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    profiler.end_video_profile("fmv_inference")

    cap.release()
    cv2.destroyAllWindows()
