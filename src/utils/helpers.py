"""Helper utilities for common tasks"""

import joblib
import json
from pathlib import Path
from typing import Any, Dict
import numpy as np


def save_model(model: Any, filepath: str) -> None:
    """
    Save model to disk using joblib
    
    Args:
        model: Model object to save
        filepath: Path where to save the model
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath: str) -> Any:
    """
    Load model from disk
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Loaded model object
    """
    return joblib.load(filepath)


def save_json(data: Dict, filepath: str) -> None:
    """
    Save dictionary as JSON file
    
    Args:
        data: Dictionary to save
        filepath: Path where to save JSON
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(filepath: str) -> Dict:
    """
    Load JSON file as dictionary
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Dictionary from JSON
    """
    with open(filepath, 'r') as f:
        return json.load(f)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types"""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def ensure_dir(directory: str) -> Path:
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        directory: Directory path
        
    Returns:
        Path object
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path
