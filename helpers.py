import os
import random
import torch
import numpy as np
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Union, List
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def set_seed(seed: int = 42) -> None:
    """
    Set all random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_logging(save_dir: str, log_level: int = logging.INFO) -> None:
    """
    Set up logging configuration with reduced verbosity during training.

    Args:
        save_dir: Directory to save log file
        log_level: Logging level for console output (default: logging.INFO)
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(save_dir, f'training_{timestamp}.log')

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Keep all logs for the file

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler (logs everything)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)

    # Console handler (logs only the specified level or higher)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(console_handler)

    # Reduce verbosity for third-party libraries
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('tqdm').setLevel(logging.ERROR)


def save_dict_to_json(dictionary: Dict, filepath: str) -> None:
    """
    Save dictionary to JSON file.

    Args:
        dictionary: Dictionary to save
        filepath: Path to save file
    """
    with open(filepath, 'w') as f:
        json.dump(dictionary, f, indent=4)

def load_dict_from_json(filepath: str) -> Dict:
    """
    Load dictionary from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)

def count_parameters(model: torch.nn.Module) -> int:
    """
    Count number of trainable parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_experiment_dir(base_dir: str, experiment_name: str) -> str:
    """
    Create directory for experiment results.

    Args:
        base_dir: Base directory
        experiment_name: Name of experiment

    Returns:
        Path to experiment directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join(base_dir, f'{experiment_name}_{timestamp}')
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir

def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device
) -> Dict:
    """
    Load model checkpoint.

    Args:
        model: Model to load checkpoint into
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Dictionary containing checkpoint information
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint

def get_device() -> torch.device:
    """
    Get available device (CPU or GPU).

    Returns:
        torch.device
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def format_time(seconds: float) -> str:
    """
    Format time in seconds to human readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

class AverageMeter:
    """Class to track running average of metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_metrics_plot(
    metrics: Dict[str, List[float]],
    save_path: str,
    title: str = "Training Metrics"
) -> None:
    """
    Save plot of training metrics.

    Args:
        metrics: Dictionary of metric lists
        save_path: Path to save plot
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    for metric_name, values in metrics.items():
        plt.plot(values, label=metric_name)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        directory: Directory path

    Returns:
        Path object of directory
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory

def save_experiment_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Save experiment configuration.

    Args:
        config: Configuration dictionary
        save_path: Path to save configuration
    """
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=4, sort_keys=True)

def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """
    Load experiment configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return json.load(f)

def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """
    Get current learning rate from optimizer.

    Args:
        optimizer: PyTorch optimizer

    Returns:
        Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def save_dataframe(df: pd.DataFrame, save_path: str) -> None:
    """
    Save DataFrame to file based on extension.

    Args:
        df: DataFrame to save
        save_path: Path to save file
    """
    ext = os.path.splitext(save_path)[1]
    if ext == '.csv':
        df.to_csv(save_path, index=False)
    elif ext == '.json':
        df.to_json(save_path, orient='records', indent=4)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")



def plot_model_comparison(
    model_metrics: Dict[str, Dict],
    save_dir: str
) -> None:
    """
    Plot comparison of different models' performance.
    
    Args:
        model_metrics: Dictionary of model names and their metrics
        save_dir: Directory to save comparison plots
    """
    metrics_to_plot = ['accuracy', 'macro_avg_f1', 'weighted_avg_f1']
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(model_metrics))
    width = 0.25
    
    for i, metric in enumerate(metrics_to_plot):
        values = [metrics[metric] for metrics in model_metrics.values()]
        plt.bar(x + i*width, values, width, label=metric)
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x + width, model_metrics.keys(), rotation=45)
    plt.legend()
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'model_comparison.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
