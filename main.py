"""
Agentic Memory Bank - Command Line Interface

Interactive mode only - simplest usage

Usage:
  python main.py                        # Start interactive mode
  python main.py --debug               # Start with debug logs
  python main.py --load memory.json    # Load memory and start
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.memory_bank import MemoryBank
from src.config import Config


def setup_logging(verbose: bool = False):
    """
    Configure logging

    Args:
        verbose: Whether to show detailed debug info

    Logging strategy:
    - Console: INFO level, shows necessary info
    - File: DEBUG level, shows all details (including all agent inputs/outputs, memory nodes, etc.)
    """
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Generate log filename (by date and time)
    log_filename = logs_dir / f"memory_bank_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Set root to DEBUG to pass all messages

    # Clear existing handlers
    root_logger.handlers.clear()

    # 1. Console Handler - INFO level (only show necessary info)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO if not verbose else logging.DEBUG)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    # 2. File Handler - DEBUG level (show all details)
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    root_logger.addHandler(file_handler)

    # Disable DEBUG logs from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("openai._base_client").setLevel(logging.WARNING)

    # Output log file location
    logging.info(f"📝 Log file: {log_filename}")



def setup_display_hook():
    """
    Setup LLM call real-time display hook

    Display strategy:
    - ReAct Agent: Show raw response (preserve all <think>, <tool_call> tags)
    - Other Agents: Show simple indicator
    - Tool internal calls: Don't show
    """
    from src.utils.llm_client import LLMClient

    # Save original method
    original_call = LLMClient.call

    def patched_call(self, messages, temperature=None, top_p=None, stop=None, max_tokens=None):
        """Intercept LLM calls, display interaction in real-time"""

        # Determine Agent type
        is_react = False
        agent_type = None

        # Handle string or list format messages
        if isinstance(messages, str):
            prompt_text = messages
            if 'intelligent assistant with memory and tool access' in prompt_text:
                is_react = True
                agent_type = "ReAct Agent"
            elif 'incremental task planning expert' in prompt_text:
                agent_type = "Planning Agent"
            elif 'topic-based clustering' in prompt_text:
                agent_type = "Classification Agent"
            elif 'information compression expert' in prompt_text:
                agent_type = "Structure Agent"
            elif 'memory relationship analysis expert' in prompt_text:
                agent_type = "Analysis Agent"
            elif 'memory integration expert' in prompt_text:
                agent_type = "Integration Agent"
            # Tool internal calls - don't show
            elif 'extract relevant information' in prompt_text.lower() or 'summarize' in prompt_text.lower():
                agent_type = None
        elif isinstance(messages, list) and len(messages) > 0:
            system_msg = messages[0].get('content', '')
            if 'intelligent assistant with memory and tool access' in system_msg:
                is_react = True
                agent_type = "ReAct Agent"
            elif 'incremental task planning expert' in system_msg:
                agent_type = "Planning Agent"
            elif 'topic-based clustering' in system_msg:
                agent_type = "Classification Agent"
            elif 'information compression expert' in system_msg:
                agent_type = "Structure Agent"
            elif 'memory relationship analysis expert' in system_msg:
                agent_type = "Analysis Agent"
            elif 'memory integration expert' in system_msg:
                agent_type = "Integration Agent"

        # Call original method
        response = original_call(self, messages, temperature, top_p, stop, max_tokens)

        # Real-time display
        if agent_type:  # Only show identified Agents
            if is_react:
                # ReAct Agent: Show raw content (preserve all tags)
                print("\n" + "=" * 80)
                print(f"🤖 {agent_type}")
                print("=" * 80)
                print(response)  # Raw output, no processing
                print("=" * 80)
            else:
                # Other Agents: Show simple indicator
                print(f"  → {agent_type} processing...")

        return response

    # Apply patch
    LLMClient.call = patched_call


def run_interactive(memory_bank: MemoryBank):
    """
    Interactive mode

    Args:
        memory_bank: MemoryBank instance
    """
    print("\n" + "=" * 100)
    print("  🚀 Agentic Memory Bank - Interactive Mode")
    print("=" * 100)
    print("\nCommands:")
    print("  Type your question directly to query")
    print("  'export <filename>' - Export memory to file")
    print("  'load <filename>' - Load memory from file")
    print("  'stats' - Show statistics")
    print("  'quit' or 'exit' - Exit program")
    print("=" * 100 + "\n")

    while True:
        try:
            # Read input
            query = input("\n> ").strip()

            if not query:
                continue

            # Handle commands
            if query.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break

            elif query.lower().startswith("export"):
                parts = query.split(maxsplit=1)
                filename = parts[1] if len(parts) > 1 else "memory_export.json"
                memory_bank.export_memory(filename)
                print(f"💾 Memory exported to: {filename}")

            elif query.lower().startswith("load"):
                parts = query.split(maxsplit=1)
                if len(parts) < 2:
                    print("Usage: load <filename>")
                    continue
                filename = parts[1]
                memory_bank.load_memory(filename)
                print(f"📥 Memory loaded from file: {filename}")

            elif query.lower() == "stats":
                print("\n📊 Current Statistics:")
                print(f"  - Graph nodes: {memory_bank.query_graph.get_node_count()}")
                print(f"  - Graph edges: {memory_bank.query_graph.get_edge_count()}")
                print(f"  - Interaction Tree entries: {memory_bank.interaction_tree.get_total_entries()}")

            else:
                # Normal query - execute directly
                print("\n" + "=" * 100)
                print("📝 User Input:")
                print("=" * 100)
                print(query)
                print("=" * 100)

                # Execute query
                result = memory_bank.run(query)

                # Show final answer
                print("\n" + "=" * 100)
                print("✅ Final Answer:")
                print("=" * 100)
                print(result["answer"])
                print("=" * 100)

                # Show statistics
                print("\n📊 Statistics:")
                stats = result["stats"]
                print(f"  - Iterations: {stats.get('iterations', 0)}")
                print(f"  - Query Graph nodes: {stats.get('graph_nodes', 0)}")
                print(f"  - Query Graph edges: {stats.get('graph_edges', 0)}")
                print(f"  - Interaction Tree entries: {stats.get('tree_entries', 0)}")
                print(f"  - Completed tasks: {stats.get('completed_tasks', 0)}")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    """Main function - Interactive mode only"""
    parser = argparse.ArgumentParser(
        description="Agentic Memory Bank - Interactive Mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage:
  python main.py --interactive          # Start interactive mode
  python main.py --interactive --debug  # Interactive mode with debug logs
  python main.py --load memory.json     # Load memory and start interactive mode
        """
    )

    # Interactive mode (always on)
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        default=True,
        help="Interactive mode (default)"
    )

    # Load memory
    parser.add_argument(
        "--load",
        type=str,
        help="Load memory from JSON file before starting"
    )

    # Debug mode
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show detailed debug logs (DEBUG level)"
    )

    args = parser.parse_args()

    # Configure logging
    setup_logging(verbose=args.debug)

    # Setup display hook
    setup_display_hook()

    try:
        # Initialize Memory Bank
        print("\n" + "=" * 100)
        print("🚀 Initializing Agentic Memory Bank...")
        print("=" * 100)
        config = Config()
        memory_bank = MemoryBank(config)
        print("✅ Initialization complete")
        print(f"  - LLM model: {config.LLM_MODEL}")
        print(f"  - Embedding model: {config.EMBEDDING_MODEL}")

        # Load memory (if specified)
        if args.load:
            print(f"\n📥 Loading memory: {args.load}")
            memory_bank.load_memory(args.load)
            print("✅ Loading complete")

        # Always run in interactive mode
        run_interactive(memory_bank)

    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
