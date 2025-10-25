# Agentic Memory Bank

A Hierarchical Graph-Based Multi-Agent System for Task-Oriented Long-Context Management

## Overview

Agentic Memory Bank is designed as a hierarchical graph-based structure managed by multiple agents. It aims to guide external agent frameworks in a task-oriented manner while structuring context. The system integrates with different external agent architectures through an Adapter, achieving a plug-and-play effect.

## Design Goal

This system is specifically designed to solve **Long-Context problems for single tasks**, rather than building a personalized Long-Term Memory Agent that learns from continuously growing memories.

## Target Scenarios

1. **DeepResearch**: Multi-source information retrieval, validation, and comprehensive analysis
2. **Long-Document QA**: Long document understanding and question answering
3. **Long Conversation/Embodied Reasoning QA**: Context management in multi-turn interactions

## Three-Layer Storage Structure

### 1. Insight Doc (Task State Layer)
Manages task execution state in a concise, structured format. Serves as default context passed to external frameworks.

### 2. Query Graph (Semantic Memory Graph Layer)
Stores structured memory summaries in graph form. Supports efficient coarse-grained retrieval through:
- Attribute-based search
- Embedding-based search

### 3. Interaction Tree (Interaction History Layer)
Stores fine-grained interaction logs in an immutable tree structure. Supports multi-modal information:
- Text branches
- Webpage branches
- Image branches
- Code branches

## Installation

### Requirements
- Python 3.10+

### Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
agentic-memory-bank/
├── core/                    # Core implementation
│   ├── models.py           # Data models
│   ├── insight_doc.py      # Insight Doc layer
│   ├── query_graph.py      # Query Graph layer
│   ├── interaction_tree.py # Interaction Tree layer
│   └── memory_manager.py   # Global manager
├── utils/                   # Utilities
│   ├── embedding.py        # Embedding computation
│   ├── serialization.py    # JSON serialization
│   └── id_generator.py     # ID generation
├── tests/                   # Unit tests
└── examples/                # Demo examples
```

## Quick Start

```python
from core import MemoryManager

# Create memory manager
manager = MemoryManager()

# Create a task
task_id = manager.create_task("Research Python async programming")

# Add memory
node_id = manager.add_memory(
    summary="Python async programming basics: coroutines and event loops",
    keywords=["python", "async", "coroutine"],
    tags=["research"],
    text_content="Detailed content about async programming..."
)

# Retrieve memories
results = manager.retrieve_memories(
    query="async event loop",
    top_k=5
)

# Save to disk
manager.save("memory_bank.json")

# Load from disk
manager = MemoryManager.load("memory_bank.json")
```

## Development Status

🚧 **Work in Progress** - Currently implementing the core three-layer storage structure.

## License

See [LICENSE](LICENSE) file for details.