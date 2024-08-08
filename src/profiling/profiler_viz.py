import matplotlib.pyplot as plt
import json
import numpy as np


def plot_system_usage(data_file: str):
    """
    Plot system usage data from a JSON file using Matplotlib.

    Args:
    data_file (str): Path to the JSON file containing system usage data.
    """
    with open(data_file, "r") as f:
        data = json.load(f)

    system_usage = data["system_usage"]
    timestamps = np.array([entry["timestamp"] for entry in system_usage])
    cpu_percent = np.array([entry["cpu_percent"] for entry in system_usage])
    memory_used_percent = np.array(
        [entry["memory_used_percent"] for entry in system_usage]
    )
    memory_used = np.array([entry["memory_used"] for entry in system_usage])
    memory_total = np.array([entry["memory_total"] for entry in system_usage])

    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    fig.suptitle("System Usage Over Time")

    # CPU
    axs[0, 0].plot(timestamps, cpu_percent, "b-")
    axs[0, 0].set_title("CPU Usage")
    axs[0, 0].set_xlabel("Timestamp")
    axs[0, 0].set_ylabel("CPU %")

    # Memory (%)
    axs[0, 1].plot(timestamps, memory_used_percent, "r-")
    axs[0, 1].set_title("Memory Usage (%)")
    axs[0, 1].set_xlabel("Timestamp")
    axs[0, 1].set_ylabel("Memory %")

    # Memory (Mb)
    axs[1, 0].plot(timestamps, memory_used, "g-")
    axs[1, 0].set_title("Memory Used")
    axs[1, 0].set_xlabel("Timestamp")
    axs[1, 0].set_ylabel("Memory Used (MB)")

    # Available Memory
    axs[1, 1].plot(timestamps, memory_total, "m-")
    axs[1, 1].set_title("Memory Total")
    axs[1, 1].set_xlabel("Timestamp")
    axs[1, 1].set_ylabel("Memory Total (MB)")

    plt.tight_layout()
    plt.show()
