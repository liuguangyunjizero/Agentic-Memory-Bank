"""
Basic CRUD Demo

Demonstrates basic Create, Read, Update, Delete operations
on the Agentic Memory Bank.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core import MemoryManager


def main():
    """Demonstrate basic CRUD operations."""
    print("=" * 60)
    print("Agentic Memory Bank - Basic CRUD Demo")
    print("=" * 60)
    print()

    # Create manager
    print("📦 Creating Memory Manager...")
    manager = MemoryManager()
    print(f"✓ Manager created\n")

    # Create task
    print("📋 Creating Task...")
    task_id = manager.create_task(
        user_question="Learn about machine learning basics",
        understood_goal="Understand fundamental ML concepts and algorithms"
    )
    print(f"✓ Task ID: {task_id}\n")

    # Add pending tasks
    print("📝 Adding Pending Tasks...")
    manager.insight_doc.add_pending_tasks([
        "Research supervised learning",
        "Study neural networks",
        "Practice with scikit-learn"
    ])
    print(f"✓ Added {len(manager.insight_doc.pending_tasks)} tasks\n")

    # CREATE: Add memories
    print("➕ CREATE: Adding Memories...")
    print("-" * 60)

    node1 = manager.add_memory(
        summary="Supervised learning uses labeled data to train models",
        keywords=["ml", "supervised", "learning", "labeled_data"],
        tags=["machine_learning", "basics"],
        text_content="In supervised learning, the algorithm learns from labeled training data..."
    )
    print(f"✓ Memory 1 added: {node1}")

    node2 = manager.add_memory(
        summary="Neural networks are composed of layers of interconnected nodes",
        keywords=["neural", "network", "layers", "nodes"],
        tags=["machine_learning", "deep_learning"],
        text_content="A neural network consists of an input layer, hidden layers, and an output layer..."
    )
    print(f"✓ Memory 2 added: {node2}")

    node3 = manager.add_memory(
        summary="Scikit-learn is a popular Python library for machine learning",
        keywords=["scikit-learn", "python", "ml", "library"],
        tags=["tools", "python"],
        webpage_content={
            "url": "https://scikit-learn.org",
            "title": "scikit-learn: machine learning in Python",
            "parsed_text": "Simple and efficient tools for predictive data analysis...",
            "links": ["https://scikit-learn.org/stable/tutorial/"]
        }
    )
    print(f"✓ Memory 3 added: {node3}\n")

    # READ: Retrieve memories
    print("🔍 READ: Retrieving Memories...")
    print("-" * 60)

    # By text query
    print("\n1. Search by text query: 'neural network'")
    results = manager.retrieve_memories(query="neural network", top_k=2)
    for i, node in enumerate(results, 1):
        print(f"   {i}. [{node.id}] {node.summary}")

    # By keywords
    print("\n2. Search by keywords: ['ml', 'learning']")
    results = manager.retrieve_memories(keywords=["ml", "learning"], top_k=3)
    for i, node in enumerate(results, 1):
        print(f"   {i}. [{node.id}] {node.summary}")

    # By tags
    print("\n3. Search by tags: ['machine_learning']")
    results = manager.retrieve_memories(tags=["machine_learning"], top_k=3)
    for i, node in enumerate(results, 1):
        print(f"   {i}. [{node.id}] {node.summary}")

    # Get full context
    print(f"\n4. Get full context for node: {node2}")
    context = manager.get_full_memory_context(node2)
    print(f"   Summary: {context['summary']}")
    print(f"   Keywords: {context['keywords']}")
    print(f"   Tags: {context['tags']}")
    print(f"   Interactions: {list(context['interactions'].keys())}\n")

    # UPDATE: Modify a memory
    print("✏️  UPDATE: Modifying Memory...")
    print("-" * 60)

    print(f"Updating summary of node {node1}...")
    manager.query_graph.update_node_summary(
        node_id=node1,
        new_summary="Supervised learning uses labeled data to train predictive models",
        reason="Added 'predictive' for clarity",
        source_interaction="manual_update"
    )
    updated_node = manager.query_graph.get_node(node1)
    print(f"✓ Updated summary: {updated_node.summary}")
    print(f"✓ Change log entries: {len(updated_node.metadata.change_log)}\n")

    # Task management
    print("📋 Task Management...")
    print("-" * 60)

    manager.insight_doc.set_current_task("Research supervised learning")
    print(f"✓ Current task: {manager.insight_doc.current_task}")

    manager.insight_doc.complete_current_task(
        result="Learned basics of supervised learning",
        impact="Ready to move to neural networks"
    )
    print(f"✓ Task completed")
    print(f"✓ Completed tasks: {len(manager.insight_doc.completed_tasks)}\n")

    # Statistics
    print("📊 Statistics...")
    print("-" * 60)

    stats = manager.get_statistics()
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Active nodes: {stats['active_nodes']}")
    print(f"Total edges: {stats['total_edges']}")
    print(f"Tree roots: {stats['tree_roots']}")
    print(f"Completed tasks: {stats['completed_tasks']}")
    print(f"Pending tasks: {stats['pending_tasks']}\n")

    # Save
    print("💾 Saving Memory Bank...")
    print("-" * 60)

    save_path = "demo_memory_bank.json"
    manager.save(save_path)
    print(f"✓ Saved to {save_path}\n")

    # Load
    print("📂 Loading Memory Bank...")
    print("-" * 60)

    loaded_manager = MemoryManager.load(save_path)
    print(f"✓ Loaded from {save_path}")

    loaded_stats = loaded_manager.get_statistics()
    print(f"✓ Verified: {loaded_stats['active_nodes']} nodes loaded\n")

    # DELETE: Clean up
    print("🗑️  DELETE: Cleaning Up...")
    print("-" * 60)

    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"✓ Removed {save_path}\n")

    print("=" * 60)
    print("✅ Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
