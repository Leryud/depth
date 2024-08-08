import cv2
import time
from src.model.loader import load_model
from src.model.inference import process_frame
from src.utils.image_processing import colorize_resize_depth_map
from src.utils.visualization import combine_images, add_labels
from src.profiling.profiler import profiler, profile


@profile
def webcam_inference(config: dict):
    """
    Run real-time inference on webcam feed.

    Args:
        config (dict): Configuration dictionary.
    """
    session = load_model(config["model"])

    cap = cv2.VideoCapture(0)

    frame_count = 0

    profiler.start_monitoring()
    profiler.start_video_profile("webcam_inference")

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        depth = process_frame(session, frame)
        # depth_map = np.asarray(depth_map, dtype=np.float32) # For fp16 inference

        orig_shape = frame.shape[:2]
        depth_color = colorize_resize_depth_map(depth, orig_shape)

        combined_frame = combine_images([frame, depth_color])
        end_time = time.time()
        frame_latency = end_time - start_time
        labeled_frame = add_labels(
            combined_frame, ["Original", f"Depth Map | Latency :{frame_latency:.4f}"]
        )

        cv2.imshow("Depth Estimation - DepthAnythingV2 ONNX", labeled_frame)

        profiler.profile_frame("webcam_inference", frame_latency)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    profiler.end_video_profile("webcam_inference")
    profiler.stop_monitoring()

    cap.release()
    cv2.destroyAllWindows()

    profiler.save("webcam_inference_profile.json")
