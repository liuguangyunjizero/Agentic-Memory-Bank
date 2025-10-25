"""
Integration Tests

Tests the complete workflow of Agentic Memory Bank.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core import MemoryManager


def test_complete_workflow():
    """Test complete workflow: create task, add memories, retrieve, save/load."""
    print("Testing complete workflow...\n")

    # 1. Create memory manager
    print("1. Creating Memory Manager...")
    manager = MemoryManager()
    print(f"   ✓ Created: {manager}\n")

    # 2. Create a task
    print("2. Creating Task...")
    task_id = manager.create_task(
        user_question="Research Python async programming",
        understood_goal="Learn about Python's asyncio library and async/await syntax"
    )
    print(f"   ✓ Task created: {task_id}\n")

    # 3. Add some pending tasks
    print("3. Adding Pending Tasks...")
    manager.insight_doc.add_pending_tasks([
        "Search for Python asyncio documentation",
        "Study async/await syntax",
        "Compare asyncio with threading"
    ])
    print(f"   ✓ Added {len(manager.insight_doc.pending_tasks)} tasks\n")

    # 4. Set current task
    print("4. Setting Current Task...")
    manager.insight_doc.set_current_task("Search for Python asyncio documentation")
    print(f"   ✓ Current task: {manager.insight_doc.current_task}\n")

    # 5. Add some memories
    print("5. Adding Memories...")

    # Memory 1: Text content
    node1_id = manager.add_memory(
        summary="Python asyncio is a library for writing concurrent code using async/await syntax",
        keywords=["python", "asyncio", "concurrent", "async"],
        tags=["programming", "python"],
        text_content="Python's asyncio library provides infrastructure for writing single-threaded concurrent code..."
    )
    print(f"   ✓ Added memory 1: {node1_id}")

    # Memory 2: Webpage content
    node2_id = manager.add_memory(
        summary="Official Python asyncio documentation explains event loops and coroutines",
        keywords=["python", "asyncio", "documentation", "event_loop"],
        tags=["documentation", "python"],
        webpage_content={
            "url": "https://docs.python.org/3/library/asyncio.html",
            "title": "asyncio — Asynchronous I/O",
            "parsed_text": "asyncio is used as a foundation for multiple Python asynchronous frameworks...",
            "links": ["https://docs.python.org/3/library/asyncio-task.html"]
        }
    )
    print(f"   ✓ Added memory 2: {node2_id}")

    # Memory 3: Code example
    node3_id = manager.add_memory(
        summary="Example of async function with await in Python",
        keywords=["python", "async", "await", "example"],
        tags=["code_example", "python"],
        code_content={
            "code": "async def main():\n    await asyncio.sleep(1)\n    print('Hello')",
            "language": "python",
            "execution_result": "Hello"
        }
    )
    print(f"   ✓ Added memory 3: {node3_id}\n")

    # 6. Complete current task
    print("6. Completing Current Task...")
    manager.insight_doc.complete_current_task(
        result="Found official documentation and basic examples",
        impact="Ready to study syntax in detail"
    )
    print(f"   ✓ Task completed\n")

    # 7. Retrieve memories
    print("7. Retrieving Memories...")

    # Search by text query
    results = manager.retrieve_memories(
        query="event loop coroutines",
        top_k=3
    )
    print(f"   ✓ Found {len(results)} memories for 'event loop coroutines'")
    for i, node in enumerate(results, 1):
        print(f"      {i}. {node.summary[:60]}...")

    # Search by keywords
    results_kw = manager.retrieve_memories(
        keywords=["python", "async"],
        top_k=5
    )
    print(f"\n   ✓ Found {len(results_kw)} memories with keywords ['python', 'async']")

    # Search by tags
    results_tag = manager.retrieve_memories(
        tags=["python"],
        top_k=5
    )
    print(f"   ✓ Found {len(results_tag)} memories with tag 'python'\n")

    # 8. Get full memory context
    print("8. Getting Full Memory Context...")
    context = manager.get_full_memory_context(node2_id)
    print(f"   ✓ Retrieved full context for node {node2_id}")
    print(f"      Summary: {context['summary'][:50]}...")
    print(f"      Interactions: {list(context['interactions'].keys())}\n")

    # 9. Get statistics
    print("9. Getting Statistics...")
    stats = manager.get_statistics()
    print(f"   ✓ Statistics:")
    print(f"      - Active nodes: {stats['active_nodes']}")
    print(f"      - Total edges: {stats['total_edges']}")
    print(f"      - Tree roots: {stats['tree_roots']}")
    print(f"      - Completed tasks: {stats['completed_tasks']}")
    print(f"      - Pending tasks: {stats['pending_tasks']}\n")

    # 10. Test InsightDoc context generation
    print("10. Testing InsightDoc Context...")
    context_str = manager.insight_doc.get_current_task_context()
    print("   ✓ Generated context for external framework:")
    print("   " + "\n   ".join(context_str.split("\n")[:10]))
    print("   ...\n")

    # 11. Save and load
    print("11. Testing Save/Load...")
    save_path = "test_memory_bank.json"

    manager.save(save_path)
    print(f"   ✓ Saved to {save_path}")

    # Load
    loaded_manager = MemoryManager.load(save_path)
    print(f"   ✓ Loaded from {save_path}")

    # Verify
    loaded_stats = loaded_manager.get_statistics()
    print(f"   ✓ Verified: {loaded_stats['active_nodes']} nodes, {loaded_stats['total_edges']} edges\n")

    # Cleanup
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"   ✓ Cleaned up test file\n")

    print("=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_complete_workflow()
