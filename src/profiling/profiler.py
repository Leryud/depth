import time
import functools
import json
from collections import defaultdict
from typing import Callable, Dict, List, Any
import psutil
import threading


class Profiler:
    """A class for profiling function calls, video processing, and system usage."""

    def __init__(self, sampling_interval: float = 1.0):
        """
        Initialize the Profiler.

        Args:
            sampling_interval (float): Interval for sampling system usage in seconds.
        """
        self.profiles: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"calls": 0.0, "total_time": 0.0, "avg_time": 0.0}
        )
        self.video_profiles: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {
                "frame_count": 0,
                "total_latency": 0.0,
                "avg_latency": 0.0,
                "fps": 0.0,
            }
        )
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.sampling_interval: float = sampling_interval
        self.system_usage: List[Dict[str, Any]] = []
        self.monitoring: bool = False
        self.monitor_thread: threading.Thread = None

    def start_monitoring(self) -> None:
        """Start monitoring system usage."""
        self.monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_system_usage)
        self.monitor_thread.start()

    def stop_monitoring(self) -> None:
        """Stop monitoring system usage."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.end_time = time.time()

    def _monitor_system_usage(self) -> None:
        """Monitor and record system usage at regular intervals."""
        while self.monitoring:
            usage = self.get_system_usage()
            usage["timestamp"] = time.time() - self.start_time
            self.system_usage.append(usage)
            time.sleep(self.sampling_interval)

    def profile(self, func: Callable) -> Callable:
        """
        Decorator for profiling function calls.

        Args:
            func (Callable): The function to be profiled.

        Returns:
            Callable: The wrapped function.
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()

            self.profiles[func.__name__]["calls"] += 1
            self.profiles[func.__name__]["total_time"] += end_time - start_time
            self.profiles[func.__name__]["avg_time"] = (
                self.profiles[func.__name__]["total_time"]
                / self.profiles[func.__name__]["calls"]
            )

            return result

        return wrapper

    def start_video_profile(self, name: str) -> None:
        """
        Start profiling a video processing task.

        Args:
            name (str): Name of the video profile.
        """
        self.start_time = time.time()
        self.video_profiles[name] = {
            "frame_count": 0,
            "total_latency": 0,
            "avg_latency": 0,
            "fps": 0,
        }

    def profile_frame(self, name: str, latency: float) -> None:
        """
        Profile a single frame in video processing.

        Args:
            name (str): Name of the video profile.
            latency (float): Latency of processing the frame.
        """
        self.video_profiles[name]["frame_count"] += 1
        self.video_profiles[name]["total_latency"] += latency

    def end_video_profile(self, name: str) -> None:
        """
        End profiling a video processing task and calculate final metrics.

        Args:
            name (str): Name of the video profile.
        """
        self.end_time = time.time()
        profile = self.video_profiles[name]
        profile["avg_latency"] = profile["total_latency"] / profile["frame_count"]
        profile["fps"] = profile["frame_count"] / (self.end_time - self.start_time)

    def get_system_usage(self) -> Dict[str, Any]:
        """
        Get current system usage statistics.

        Returns:
            Dict[str, Any]: Dictionary containing CPU and memory usage information.
        """
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()

        return {
            "cpu_percent": cpu_percent,
            "memory_used_percent": memory.percent,
            "memory_used": memory.used >> 20,
            "memory_total": memory.total >> 20,
        }

    def to_json(self) -> str:
        """
        Convert profiling data to JSON format.

        Returns:
            str: JSON string of profiling data.
        """
        return json.dumps(
            {
                "function_profiles": dict(self.profiles),
                "video_profiles": dict(self.video_profiles),
                "system_usage": self.system_usage,
            },
            indent=2,
        )

    def save(self, filepath: str) -> None:
        """
        Save profiling data to a JSON file.

        Args:
            filepath (str): Path to save the JSON file.
        """
        with open(filepath, "w") as f:
            f.write(self.to_json())


profiler = Profiler()


def profile(func: Callable) -> Callable:
    """
    Decorator for function profiling using the global profiler instance.

    Args:
        func (Callable): The function to be profiled.

    Returns:
        Callable: The wrapped function.
    """
    return profiler.profile(func)
