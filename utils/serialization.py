"""
Serialization Module

Provides utilities for saving and loading data to/from JSON files,
and converting between numpy arrays and lists.
"""

import json
import numpy as np
from typing import Any, Dict
from pathlib import Path


def numpy_to_list(arr: np.ndarray) -> list:
    """
    Convert numpy array to list for JSON serialization.

    Args:
        arr: Numpy array

    Returns:
        list: Python list
    """
    if arr is None:
        return None
    return arr.tolist()


def list_to_numpy(lst: list) -> np.ndarray:
    """
    Convert list to numpy array.

    Args:
        lst: Python list

    Returns:
        np.ndarray: Numpy array, or None if input is None
    """
    if lst is None:
        return None
    return np.array(lst)


def save_to_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save data to JSON file.

    Args:
        data: Dictionary to save
        filepath: Path to save the JSON file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_from_json(filepath: str) -> Dict[str, Any]:
    """
    Load data from JSON file.

    Args:
        filepath: Path to the JSON file

    Returns:
        dict: Loaded data

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
