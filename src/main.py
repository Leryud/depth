import argparse
from src.utils.config import get_config
from src.applications.cam_inference import webcam_inference
from src.applications.fmv_inference import fmv_inference
from src.applications.img_inference import img_inference
from src.profiling.profiler import profiler
from src.profiling.profiler_viz import plot_system_usage
from src.eval.eval import run_eval


def main():
    parser = argparse.ArgumentParser(description="Depth Estimation")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["cam", "fmv", "img", "evaluate"],
        required=True,
        help="Mode of operation: webcam or evaluate",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/example_fmv.mp4",
        help="Path to the input video or image file.",
    )
    parser.add_argument(
        "--eval-output",
        type=str,
        default="src/eval/result/output.json",
        help="Path to save the metrics results",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    parser.add_argument(
        "--profile-output",
        type=str,
        default="src/profiling/result/profile_output.json",
        help="Path to save profiling results",
    )
    parser.add_argument(
        "--plot-profile", action="store_true", help="Plot system usage after profiling"
    )

    args = parser.parse_args()

    config = get_config(args.config)

    if args.profile:
        profiler.start_monitoring()

    if args.mode == "cam":
        webcam_inference(config)

    if args.mode == "fmv":
        fmv_inference(config, args.input)

    if args.mode == "img":
        img_inference(config, args.input)

    elif args.mode == "evaluate":
        metrics = run_eval(args.eval_output)
        print("Evaluation Metrics:")
        print(metrics.get_value())

    if args.mode != "img":
        # We don't profile for single image inference
        if args.profile:
            profiler.stop_monitoring()
            profiler.save(args.profile_output)
            print(f"Profiling results saved to {args.profile_output}")

            if args.plot_profile:
                plot_system_usage(args.profile_output)
                print(f"System usage plot generated for {args.profile_output}")


if __name__ == "__main__":
    main()
