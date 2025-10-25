"""
ID Generator Module

Generates unique IDs for different entities in the memory bank.
"""

import uuid
from datetime import datetime


def generate_node_id() -> str:
    """
    Generate a unique ID for Query Graph nodes.

    Format: node_<timestamp>_<uuid_suffix>
    Example: node_20240115_a3f5b2

    Returns:
        str: Unique node ID
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    uuid_suffix = str(uuid.uuid4())[:6]
    return f"node_{timestamp}_{uuid_suffix}"


def generate_tree_id() -> str:
    """
    Generate a unique ID for Interaction Tree nodes.

    Format: tree_<timestamp>_<uuid_suffix>
    Example: tree_20240115_b7c8d1

    Returns:
        str: Unique tree node ID
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    uuid_suffix = str(uuid.uuid4())[:6]
    return f"tree_{timestamp}_{uuid_suffix}"


def generate_task_id() -> str:
    """
    Generate a unique ID for tasks.

    Format: task_<timestamp>_<uuid_suffix>
    Example: task_20240115_e9f0a2

    Returns:
        str: Unique task ID
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    uuid_suffix = str(uuid.uuid4())[:6]
    return f"task_{timestamp}_{uuid_suffix}"
