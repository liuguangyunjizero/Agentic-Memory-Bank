"""
Agentic Memory Bank - Command Line Interface

Provides interactive mode for querying with persistent memory management.
"""

import sys
import argparse
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Any, Optional

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.memory_bank import MemoryBank
from src.config import Config

# Agent identifier keywords for display hooks
AGENT_IDENTIFIERS = {
    "ReAct Agent": "Web Information Seeking Master with memory and tool access",
    "Planning Agent": "incremental task planning expert",
    "Classification Agent": "topic-based clustering",
    "Structure Agent": "information compression expert",
    "Analysis Agent": "memory relationship analysis expert",
    "Integration Agent": "memory integration expert"
}

# Third-party library loggers to suppress
SUPPRESSED_LOGGERS = [
    "httpx", "httpcore", "sentence_transformers",
    "urllib3", "urllib3.connectionpool", "openai", "openai._base_client"
]


def setup_logging() -> None:
    """Configure logging with console (INFO) and file (DEBUG) output."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    log_filename = logs_dir / f"memory_bank_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    for logger_name in SUPPRESSED_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    logging.info(f"📝 Log file: {log_filename}")


def _detect_agent_type(messages: Any) -> tuple[Optional[str], bool]:
    """
    Detect agent type from LLM messages.
    Returns (agent_name, is_react_agent) or (None, False) for tool-internal calls.
    """
    prompt_text = ""
    if isinstance(messages, str):
        prompt_text = messages
    elif isinstance(messages, list) and len(messages) > 0:
        prompt_text = messages[0].get('content', '')

    if not prompt_text:
        return None, False

    for agent_name, identifier in AGENT_IDENTIFIERS.items():
        if identifier in prompt_text:
            is_react = (agent_name == "ReAct Agent")
            return agent_name, is_react

    if 'extract relevant information' in prompt_text.lower() or 'summarize' in prompt_text.lower():
        return None, False

    return None, False


def setup_display_hook() -> None:
    """
    Monkey-patch LLMClient.call to display agent activity.
    Shows simple indicators for non-ReAct agents (ReAct handles its own display).
    """
    from src.utils.llm_client import LLMClient

    original_call = LLMClient.call

    def patched_call(
        self,
        messages: Any,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Any = None,
        max_tokens: Optional[int] = None
    ) -> str:
        agent_type, is_react = _detect_agent_type(messages)
        response = original_call(self, messages, temperature, top_p, stop, max_tokens)

        if agent_type and not is_react:
            print(f"  → {agent_type} processing...")

        return response

    LLMClient.call = patched_call


def run_interactive(memory_bank: MemoryBank) -> None:
    """
    Interactive query mode with context loading and memory management.
    Supports two modes: direct questions and context-only loading.
    """
    print("\n🚀 Agentic Memory Bank - Interactive Mode")
    print("\n📖 Input Modes:")
    print("  1. Direct question: Just type your question")
    print("  2. Load context: Start with 'Context:' or '上下文：'")
    print("\n⚙️  Commands:")
    print("  - 'export <file>': Save memory to file")
    print("  - 'load <file>': Load memory from file")
    print("  - 'clear': Clear all memory")
    print("  - 'quit': Exit the program\n")

    context_loaded = False

    while True:
        try:
            if context_loaded:
                prompt = "✓ > "
            else:
                prompt = "> "

            query = input(prompt).strip()
            if not query:
                continue

            if query.lower() in ["quit", "exit"]:
                print("\n🧹 Cleaning up memory...")
                memory_bank.clear_memory()
                print("Goodbye!")
                break

            elif query.lower().startswith("export"):
                parts = query.split(maxsplit=1)
                filename = parts[1] if len(parts) > 1 else "memory_export.json"
                try:
                    memory_bank.export_memory(filename)
                    print(f"💾 Exported to: {filename}")
                except Exception as e:
                    print(f"❌ Export failed: {e}")

            elif query.lower().startswith("load"):
                parts = query.split(maxsplit=1)
                if len(parts) < 2:
                    print("Usage: load <filename>")
                    continue
                filename = parts[1]
                try:
                    memory_bank.load_memory(filename)
                    print(f"📥 Loaded from: {filename}")
                    context_loaded = False
                except FileNotFoundError:
                    print(f"❌ File not found: {filename}")
                except Exception as e:
                    print(f"❌ Load failed: {e}")

            elif query.lower() == "clear":
                confirm = input("⚠️  Clear all memory? This cannot be undone. (yes/no): ").strip().lower()
                if confirm == "yes":
                    memory_bank.clear_memory()
                    print("✅ Memory cleared")
                    context_loaded = False
                else:
                    print("Cancelled")

            else:
                is_context_only = query.lower().startswith("context:") or query.startswith("上下文：")

                if is_context_only:
                    print(f"\n📚 Loading context...")
                elif context_loaded and not ("context:" in query.lower() or "上下文：" in query):
                    print(f"\n💡 Using previously loaded context to answer...")
                else:
                    print(f"\n📝 Query: {query}")

                result = memory_bank.run(query)

                if is_context_only:
                    context_loaded = True
                    print("\n✅ Context loaded successfully. You can now ask questions.")

                print()

        except KeyboardInterrupt:
            print("\n\n🧹 Cleaning up memory...")
            memory_bank.clear_memory()
            print("Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            traceback.print_exc()


def main() -> None:
    """Main entry point for interactive mode with optional memory loading."""
    parser = argparse.ArgumentParser(
        description="Agentic Memory Bank - Interactive Mode"
    )

    parser.add_argument(
        "--load",
        type=str,
        help="Load memory from JSON file before starting"
    )

    args = parser.parse_args()

    setup_logging()
    setup_display_hook()

    try:
        print("\n🚀 Initializing Agentic Memory Bank...")
        config = Config()
        memory_bank = MemoryBank(config)
        print(f"✅ Ready | Model: {config.LLM_MODEL} | Embedding: {config.EMBEDDING_MODEL}")

        if args.load:
            try:
                if not Path(args.load).exists():
                    print(f"⚠️  Warning: File not found: {args.load}")
                else:
                    memory_bank.load_memory(args.load)
                    print(f"📥 Loaded memory from: {args.load}")
            except Exception as e:
                print(f"⚠️  Warning: Failed to load memory: {e}")

        run_interactive(memory_bank)

    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()