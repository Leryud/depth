import numpy as np
import onnxruntime as ort

from src.depth_anything.util.transform import load_image
from src.profiling.profiler import profile


@profile
def process_frame(
    session: ort.InferenceSession, frame: np.ndarray, load: bool = True
) -> np.ndarray:
    """
    Process a single frame through the ONNX inference session to estimate depth.

    Args:
        session (ort.InferenceSession): The ONNX runtime inference session.
        frame (Union[str, np.ndarray]): A numpy array representing the image.
        load (bool, optional): If True, applies DepthAnythingV2 transforms. If False, assume frame is ready for inference. Defaults to True.

    Returns:
        np.ndarray: The estimated depth map.
    """
    if load:
        image, (orig_h, orig_w) = load_image(frame)
    else:
        image = frame
        orig_h, orig_w = image.shape[:2]

    binding = session.io_binding()
    ort_input = session.get_inputs()[0].name
    binding.bind_cpu_input(ort_input, image)
    ort_output = session.get_outputs()[0].name
    binding.bind_output(ort_output, "cpu")

    session.run_with_iobinding(binding)

    depth = binding.get_outputs()[0].numpy()

    return depth.astype(np.float32)
