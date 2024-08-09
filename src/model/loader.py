from src.profiling.profiler import profile
import onnxruntime as ort


@profile
def load_model(model_path: str) -> ort.InferenceSession:
    """
    Load the MiDaS model.

    Args:
        device (str): Device to load the model on ('cpu', 'cuda', or 'mps').
        model_type (str): Type of the MiDaS model.
        model_path (str): Path to the model weights.

    Returns:
        Tuple[torch.nn.Module, Any, int, int]: Loaded model, transform function, height, and width.
    """
    return ort.InferenceSession(
        model_path,
        providers=[
            # "CUDAExecutionProvider",
            "CPUExecutionProvider",
            # "CoreMLExecutionProvider",
        ],
    )
