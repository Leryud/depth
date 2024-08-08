# Depth Estimation

## Project Overview

This project implements real-time depth estimation using the DepthAnythingV2 model in its ONNX version. It provides functionality for webcam-based real-time inference and file-based (image and video) inference. The application is designed with modularity, extensibility, and performance profiling in mind.
The model used in this project is evaluated on the NYU Depth V2 dataset

## Project Structure

```bash
src/
├── config # YAML configuration files
├── data
│   └── nyu2_test # Dataset used for evaluation
├── src
│   ├── applications # Inference files
│   ├── dataset # Dataset loader and preprocessing functions
│   ├── depth_anything # Off-the-shelf model
│   ├── depth_anything_v2 # Off-the-shelf model
│   ├── eval # Evalutation pipeline
│   │   └── result # Evaluation results
│   ├── model # Model utility functions, loading and inference
│   ├── optim # Conversion to FP16 of the ONNX model *[1]
│   ├── profiling # Profiling class and visualising
│   │   └── result # Profiling results
│   └── utils # Utility files for config, image processing, viz
└── weights
```

## Features

- Real-time depth estimation using webcam input
- Image and video file-based depth estimation
- Performance profiling and visualization
- Model evaluation on NYU Depth V2 dataset
- Support for FP16 and FP32 ONNX models
- Configurable settings via YAML files

## Installation



## Usage

The application provides a command-line interface (CLI) for easy use. Here are some example commands:

```bash
# Run webcam inference
python main.py --mode cam

# Run evaluation
python main.py --mode evaluate --metrics-output ./evaluation_results

# Run webcam inference with profiling and plot
python main.py --mode cam --profile --plot-profile

# Run evaluation with a custom config file and custom profiling output
python main.py --mode evaluate --config custom_config.yaml --metrics-output ./evaluation_results --profile --profile-output custom_profile.json
```

### Command-line Arguments

- `--mode`: Choose between ' img', 'fmv', 'cam' and 'evaluate' modes (required)
- `--input`: The input path to a given image or video, for 'img' and 'fmv' modes
- `--config`: Path to the configuration file (default: 'config/config.yaml')
- `--profile`: Enable profiling
- `--profile-output`: Path to save profiling results
- `--plot-profile`: Plot system usage after profiling
- `--eval-output`: Directory to save evaluation metrics JSON file (required for evaluate mode)

## Evaluation

The project includes an evaluation pipeline for the NYU Depth V2 dataset.
Here is the link to the test dataset I used: [NYU Depth 2 test split](https://www.dropbox.com/scl/fo/0k0gciat7cyc9fvq8lof0/ANgqcVVAdRIJIINpXieCh9M?rlkey=2uldkkr7pyvnoezjroh6vl2lj&st=mp0lf0q5&dl=0)
Simply place it into the `data` folder, and run the evaluation like so :

```bash
python main.py --mode evaluate --metrics-output evaluation_results.json
```

This will generate a JSON file with evaluation metrics in the specified output file.

## Profiling

The application includes a built-in profiler for performance analysis. To enable profiling, use the `--profile` flag. The results can be visualized using the `--plot-profile` flag.

## Model Optimization

The project supports both FP32 and FP16 ONNX models. FP16 models can potentially offer improved performance, especially on devices with hardware acceleration for half-precision operations. However, performance vary depending on the specific hardware.
I have found that it does not improve inference speed on my M1 chip, but seems to consume less RAM and CPU.

## Known Issues

## Future Improvements
