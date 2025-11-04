"""
Agentic Memory Bank - 命令行接口

简化版：默认显示完整的ReAct交互过程

用法：
  python main.py "Among CS conferences, in 2025, which conference has..."
  python main.py --file input.txt
  python main.py --interactive
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
    配置日志

    Args:
        verbose: 是否显示详细调试信息
    """
    level = logging.DEBUG if verbose else logging.WARNING

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 始终禁用第三方库的INFO日志
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


def setup_display_hook():
    """
    设置LLM调用的实时显示钩子

    显示策略：
    - ReAct Agent: 显示原始响应（保留所有<think>、<tool_call>等标签）
    - 其他Agent: 只显示简单提示
    - 工具内部调用: 不显示
    """
    from src.utils.llm_client import LLMClient

    # 保存原始方法
    original_call = LLMClient.call

    def patched_call(self, messages, temperature=None, top_p=None, stop=None, max_tokens=None):
        """拦截LLM调用，实时显示交互"""

        # 判断Agent类型
        is_react = False
        agent_type = None

        # 处理字符串或列表格式的messages
        if isinstance(messages, str):
            prompt_text = messages
            if '你是一个拥有记忆和工具访问能力的智能助手' in prompt_text:
                is_react = True
                agent_type = "ReAct Agent"
            elif '任务规划专家' in prompt_text:
                agent_type = "Planning Agent"
            elif '上下文分类专家' in prompt_text:
                agent_type = "Classification Agent"
            elif '信息结构化专家' in prompt_text:
                agent_type = "Structure Agent"
            elif '记忆关系分析专家' in prompt_text:
                agent_type = "Analysis Agent"
            elif '记忆整合专家' in prompt_text:
                agent_type = "Integration Agent"
            # 工具内部调用 - 不显示
            elif 'extract relevant information' in prompt_text.lower() or 'summarize' in prompt_text.lower():
                agent_type = None
        elif isinstance(messages, list) and len(messages) > 0:
            system_msg = messages[0].get('content', '')
            if '你是一个拥有记忆和工具访问能力的智能助手' in system_msg:
                is_react = True
                agent_type = "ReAct Agent"
            elif '任务规划专家' in system_msg:
                agent_type = "Planning Agent"
            elif '上下文分类专家' in system_msg:
                agent_type = "Classification Agent"
            elif '信息结构化专家' in system_msg:
                agent_type = "Structure Agent"
            elif '记忆关系分析专家' in system_msg:
                agent_type = "Analysis Agent"
            elif '记忆整合专家' in system_msg:
                agent_type = "Integration Agent"

        # 调用原始方法
        response = original_call(self, messages, temperature, top_p, stop, max_tokens)

        # 实时显示
        if agent_type:  # 只显示已识别的Agent
            if is_react:
                # ReAct Agent：显示原始内容（保留所有标签）
                print("\n" + "─" * 100)
                print(f"🤖 {agent_type}")
                print("─" * 100)
                print(response)  # 原始输出，不加工
                print("─" * 100)
            else:
                # 其他Agent：只显示简单提示
                print(f"\n🤖 正在调用 {agent_type}...")

        return response

    # 应用patch
    LLMClient.call = patched_call


def run_query(memory_bank: MemoryBank, query: str, output_file: str = None):
    """
    运行查询

    Args:
        memory_bank: MemoryBank实例
        query: 查询字符串
        output_file: 输出文件路径（可选）
    """
    print("\n" + "=" * 100)
    print("📝 用户输入:")
    print("=" * 100)
    print(query)
    print("=" * 100)

    # 执行查询
    result = memory_bank.run(query)

    # 显示最终答案
    print("\n" + "=" * 100)
    print("✅ 最终答案:")
    print("=" * 100)
    print(result["answer"])
    print("=" * 100)

    # 显示统计
    print("\n📊 统计信息:")
    stats = result["stats"]
    print(f"  - 执行轮次: {stats.get('iterations', 0)}")
    print(f"  - Query Graph节点: {stats.get('graph_nodes', 0)}")
    print(f"  - Query Graph边: {stats.get('graph_edges', 0)}")
    print(f"  - Interaction Tree条目: {stats.get('tree_entries', 0)}")
    print(f"  - 已完成任务: {stats.get('completed_tasks', 0)}")

    # 导出记忆（如果指定）
    if output_file:
        memory_bank.export_memory(output_file)
        print(f"\n💾 记忆已导出到: {output_file}")


def run_interactive(memory_bank: MemoryBank):
    """
    交互式模式

    Args:
        memory_bank: MemoryBank实例
    """
    print("\n" + "=" * 100)
    print("  🚀 Agentic Memory Bank - 交互式模式")
    print("=" * 100)
    print("\n命令:")
    print("  直接输入问题进行查询")
    print("  'export <文件名>' - 导出记忆")
    print("  'load <文件名>' - 加载记忆")
    print("  'stats' - 显示统计信息")
    print("  'quit' 或 'exit' - 退出")
    print("=" * 100 + "\n")

    while True:
        try:
            # 读取输入
            query = input("\n> ").strip()

            if not query:
                continue

            # 处理命令
            if query.lower() in ["quit", "exit"]:
                print("再见！")
                break

            elif query.lower().startswith("export"):
                parts = query.split(maxsplit=1)
                filename = parts[1] if len(parts) > 1 else "memory_export.json"
                memory_bank.export_memory(filename)
                print(f"💾 记忆已导出到: {filename}")

            elif query.lower().startswith("load"):
                parts = query.split(maxsplit=1)
                if len(parts) < 2:
                    print("用法: load <文件名>")
                    continue
                filename = parts[1]
                memory_bank.load_memory(filename)
                print(f"📥 记忆已从文件加载: {filename}")

            elif query.lower() == "stats":
                print("\n📊 当前统计:")
                print(f"  - Graph节点数: {memory_bank.query_graph.get_node_count()}")
                print(f"  - Graph边数: {memory_bank.query_graph.get_edge_count()}")
                print(f"  - Interaction Tree条目数: {memory_bank.interaction_tree.get_total_entries()}")

            else:
                # 普通查询
                run_query(memory_bank, query)

        except KeyboardInterrupt:
            print("\n\n中断。再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Agentic Memory Bank - 代理式记忆银行",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py "Among CS conferences, in 2025, which conference has..."
  python main.py --file input.txt
  python main.py --interactive
  python main.py --interactive --debug  (显示详细调试信息)
        """
    )

    # 输入方式
    parser.add_argument(
        "query",
        nargs="?",
        type=str,
        help="查询字符串（如果不指定则进入交互模式）"
    )

    parser.add_argument(
        "--file", "-f",
        type=str,
        help="从文件读取查询"
    )

    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="交互式模式"
    )

    # 输出选项
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="输出记忆到JSON文件"
    )

    # 加载记忆
    parser.add_argument(
        "--load",
        type=str,
        help="从JSON文件加载记忆"
    )

    # 调试选项
    parser.add_argument(
        "--debug",
        action="store_true",
        help="显示详细的调试信息（DEBUG级别日志）"
    )

    args = parser.parse_args()

    # 如果没有指定任何输入，自动进入交互模式
    if not args.query and not args.file and not args.interactive:
        args.interactive = True

    # 配置日志
    setup_logging(verbose=args.debug)

    # 设置实时显示钩子
    setup_display_hook()

    try:
        # 初始化Memory Bank
        print("\n" + "=" * 100)
        print("🚀 初始化 Agentic Memory Bank...")
        print("=" * 100)
        config = Config()
        memory_bank = MemoryBank(config)
        print("✅ 初始化完成")
        print(f"  - LLM模型: {config.LLM_MODEL}")
        print(f"  - Embedding模型: {config.EMBEDDING_MODEL}")

        # 加载记忆（如果指定）
        if args.load:
            print(f"\n📥 加载记忆: {args.load}")
            memory_bank.load_memory(args.load)
            print("✅ 加载完成")

        # 执行模式
        if args.interactive:
            run_interactive(memory_bank)
        elif args.file:
            with open(args.file, 'r', encoding='utf-8') as f:
                query = f.read()
            run_query(memory_bank, query, args.output)
        elif args.query:
            run_query(memory_bank, args.query, args.output)

    except Exception as e:
        print(f"\n❌ 致命错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
