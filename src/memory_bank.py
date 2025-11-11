"""
Agentic Memory Bank 核心类

整合所有组件，提供完整的记忆管理功能。
"""

import logging
import uuid
import time
from typing import Dict, Any, List, Optional, Tuple

from src.storage.insight_doc import InsightDoc, CompletedTask, TaskType
from src.storage.query_graph import QueryGraph, QueryGraphNode
from src.storage.interaction_tree import InteractionTree
from src.modules.embedding import EmbeddingModule
from src.modules.retrieval import RetrievalModule
from src.modules.graph_ops import GraphOperations
from src.agents.classification_agent import ClassificationAgent, ClassificationInput
from src.agents.structure_agent import StructureAgent, StructureInput
from src.agents.analysis_agent import AnalysisAgent, AnalysisInput
from src.agents.integration_agent import IntegrationAgent, IntegrationInput, NodeWithNeighbors
from src.agents.planning_agent import PlanningAgent, PlanningInput, ConflictNotification
from src.interface.adapter import MemoryBankAdapter
from src.tools.deep_retrieval_tool import DeepRetrievalTool
from src.tools.search_tool import SearchTool
from src.tools.visit_tool import VisitTool
from src.tools.react_agent import MultiTurnReactAgent
from src.utils.llm_client import LLMClient
from src.utils.file_utils import FileUtils
from src.config import Config
from src.prompts.agent_prompts import REACT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class MemoryBank:
    """Agentic Memory Bank主类"""

    def __init__(self, config: Config = None):
        """
        初始化Memory Bank

        Args:
            config: 配置对象（可选，不提供则使用默认配置）
        """
        logger.info("Initializing Agentic Memory Bank...")

        # 初始化配置
        self.config = config or Config()

        # 初始化存储层
        self.query_graph = QueryGraph()
        self.interaction_tree = InteractionTree()
        self.insight_doc = None  # 每次任务单独创建

        # 初始化工具
        self.llm_client = LLMClient.from_config(self.config)
        self.file_utils = FileUtils(
            temp_dir=self.config.TEMP_DIR,
            storage_dir=self.config.STORAGE_DIR
        )

        # 初始化硬编码模块
        self.embedding_module = EmbeddingModule.from_config(self.config)
        self.retrieval_module = RetrievalModule(
            alpha=self.config.RETRIEVAL_ALPHA,
            k=self.config.RETRIEVAL_K
        )
        self.graph_ops = GraphOperations(self.query_graph)

        # 初始化Agent
        self.classification_agent = ClassificationAgent.from_config(
            self.llm_client, self.config
        )
        self.structure_agent = StructureAgent.from_config(self.llm_client, self.config)
        self.analysis_agent = AnalysisAgent.from_config(self.llm_client, self.config)
        self.integration_agent = IntegrationAgent.from_config(self.llm_client, self.config)
        self.planning_agent = PlanningAgent.from_config(self.llm_client, self.config)

        # 初始化Interface层
        self.deep_retrieval_tool = DeepRetrievalTool(self.interaction_tree, self.file_utils)

        # 搜索工具：如果配置了Serper API key，使用真实搜索
        search_api_key = self.config.SERPER_API_KEY
        if not search_api_key or search_api_key == "your-serper-api-key-here":
            raise ValueError(
                "未配置Serper API key。请在.env文件中设置SERPER_API_KEY。\n"
                "注册地址：https://serper.dev/"
            )

        self.search_tool = SearchTool(search_api_key=search_api_key)

        # Visit工具：使用Jina Reader API（必需）
        jina_api_key = self.config.JINA_API_KEY
        self.visit_tool = VisitTool(
            llm_client=self.llm_client,
            jina_api_key=jina_api_key,
            temperature=self.config.VISIT_EXTRACTION_TEMPERATURE,
            top_p=self.config.VISIT_EXTRACTION_TOP_P
        )

        # 初始化Adapter
        self.adapter = MemoryBankAdapter(self, self.retrieval_module, self.file_utils)

        # 初始化ReAct Agent
        tools = {
            "deep_retrieval": self.deep_retrieval_tool,
            "search": self.search_tool,
            "visit": self.visit_tool
        }
        self.react_agent = MultiTurnReactAgent(
            llm_client=self.llm_client,
            tools=tools,
            system_message=REACT_SYSTEM_PROMPT,
            max_iterations=self.config.MAX_LLM_CALL_PER_RUN,
            max_context_tokens=self.config.MAX_CONTEXT_TOKENS,
            temperature=self.config.REACT_AGENT_TEMPERATURE,
            top_p=self.config.REACT_AGENT_TOP_P
        )

        logger.info("Agentic Memory Bank initialized successfully")

    def run(self, user_input: str) -> Dict[str, Any]:
        """
        执行单次任务

        Args:
            user_input: 用户输入（可能包含上下文+问题）

        Returns:
            任务结果：{
                "answer": str,
                "insight_doc": dict,
                "stats": dict
            }
        """
        logger.info("=" * 60)
        logger.info(f"Starting new task: {user_input[:300]}{'...' if len(user_input) > 300 else ''}")
        logger.info("=" * 60)

        try:
            # 1. 初始化阶段
            print("\n" + "=" * 80)
            print("  Memory Bank Initialized")
            print("=" * 80)
            enhanced_prompt = self._initialize(user_input)
            iterations = 0

            # 显示任务目标和初始状态
            if self.insight_doc:
                print(f"Task: {self.insight_doc.task_goal}")
                print(f"Current Subtask: {self.insight_doc.current_task if self.insight_doc.current_task else '(none)'}")
                print("=" * 80)

            # 2. 执行循环
            answer = None
            last_react_result = None  # 保存最后一次ReAct结果
            max_iterations = self.config.MAX_LLM_CALL_PER_RUN
            while not self._should_terminate() and iterations < max_iterations:
                iterations += 1
                print(f"\n{'=' * 80}")
                print(f"  Iteration {iterations}")
                print("=" * 80)

                # 显示当前任务状态（简化）
                if self.insight_doc and self.insight_doc.current_task:
                    print(f"Current: {self.insight_doc.current_task}")
                    print(f"Completed: {len(self.insight_doc.completed_tasks)} | Memory nodes: {self.query_graph.get_node_count()}")

                # 2.1 ReAct执行
                react_result = self.react_agent.run(enhanced_prompt)
                last_react_result = react_result  # 保存结果
                answer = react_result.get("prediction", "")

                # 2.2 上下文拦截：提取搜索结果并转化为记忆
                # 检查是否是最终回答任务
                current_task = self.insight_doc.current_task if self.insight_doc else ""
                is_final_answer_task = "根据现有相关记忆直接回答问题" in current_task or "根据现有记忆回答问题" in current_task

                if self.insight_doc and self.insight_doc.current_task:
                    # 检查是否有工具调用（通过检查消息历史）
                    messages = react_result.get("messages", [])
                    has_tool_calls = any(
                        msg.get("role") == "assistant" and "<tool_call>" in msg.get("content", "")
                        for msg in messages
                    )

                    tool_responses = self._extract_tool_responses(messages)

                    if has_tool_calls and not is_final_answer_task:
                        # 有工具调用，且不是最终回答任务 - 进入记忆处理流程
                        # (No console output - details logged to file)

                        # 提取完整上下文（包括思考过程、工具调用和工具响应）
                        # Classification Agent 需要完整上下文来准确分类和理解推理过程
                        full_context = self._extract_full_context(messages)

                        # 调用上下文拦截机制
                        try:
                            # 判断任务类型（参考 REQUIREMENTS_FINAL.md 第4.2节）
                            has_conflict = self.adapter.has_pending_conflicts()
                            if has_conflict:
                                task_type = "CROSS_VALIDATE"
                            else:
                                task_type = "NORMAL"

                            # intercept_context 内部会调用 Planning Agent 更新 insight_doc
                            # 包括 completed_tasks 和 pending_tasks
                            self.adapter.intercept_context(full_context, task_type, self.insight_doc)
                            # (Memory processing complete - logged to file)

                        except Exception as e:
                            print(f"Error: {str(e)}")
                            logger.error(f"Context interception failed: {str(e)}")
                            import traceback
                            traceback.print_exc()

                    elif is_final_answer_task and react_result.get("termination") == "answer":
                        # 最终回答任务且有答案 - 但不能直接认为完成，需要Planning Agent验证
                        answer = react_result.get("prediction", "")

                        # 调用Planning Agent验证答案
                        from src.agents.planning_agent import PlanningInput
                        planning_output = self.planning_agent.run(PlanningInput(
                            insight_doc=self.insight_doc,
                            new_memory_nodes=[
                                {
                                    "id": "final_answer",
                                    "context": "Final answer candidate",
                                    "keywords": ["answer", "final"],
                                    "summary": answer
                                }
                            ],
                            conflict_notification=None
                        ))

                        # 更新insight_doc
                        self.insight_doc.task_goal = planning_output.task_goal
                        self.insight_doc.completed_tasks = planning_output.completed_tasks
                        self.insight_doc.current_task = planning_output.current_task

                        # 如果Planning Agent判断没有后续任务了，才真正结束
                        if not self.insight_doc.current_task:
                            print("\n[DONE] Task complete - Final answer obtained and verified")
                            break

                    elif not tool_responses:
                        # 没有工具调用，但ReAct返回了答案 - 直接标记任务完成
                        # (ReAct provided direct answer without tools)

                        if self.insight_doc.current_task:
                            # 保存答案到外层变量（从prediction字段获取）
                            answer = react_result.get("prediction", "")

                            # 标记任务完成
                            self.insight_doc.current_task = ""
                            completed_task = CompletedTask(
                                type=TaskType.NORMAL,
                                description=current_task,
                                status="Success",
                                context=f"直接回答: {answer[:200]}"
                            )
                            self.insight_doc.completed_tasks.append(completed_task)

                            # 如果是最终回答任务，直接结束循环
                            if is_final_answer_task:
                                print("\n[DONE] Task complete - Final answer obtained")
                                break

                            # 调用Planning Agent检查是否还有其他任务
                            from src.agents.planning_agent import PlanningInput
                            planning_output = self.planning_agent.run(PlanningInput(
                                insight_doc=self.insight_doc,
                                new_memory_nodes=[],  # 没有生成记忆节点
                                conflict_notification=None
                            ))

                            # 更新insight_doc
                            self.insight_doc.current_task = planning_output.current_task

                # 2.3 检查是否有待办任务（如果没有，说明任务完成）
                if not self.insight_doc or not self.insight_doc.current_task:
                    print("\n[DONE] All tasks complete")
                    break

                # 2.4 如果ReAct已经给出答案且无更多待办任务，结束
                if react_result.get("termination") == "answer" and not self.insight_doc.current_task:
                    print("\n[DONE] Task complete")
                    break

                # 2.5 增强下一轮Prompt（基于新的任务状态）
                if self.insight_doc.current_task:
                    print(f"\nNext: {self.insight_doc.current_task}")
                enhanced_prompt = self.adapter.enhance_prompt(self.insight_doc)

            # 3. 统计信息
            final_insight_doc = self.insight_doc.to_dict() if self.insight_doc else None

            stats = {
                "iterations": iterations,
                "graph_nodes": self.query_graph.get_node_count(),
                "graph_edges": self.query_graph.get_edge_count(),
                "tree_entries": self.interaction_tree.get_total_entries(),
                "completed_tasks": len(self.insight_doc.completed_tasks) if self.insight_doc else 0,
                "current_task": self.insight_doc.current_task if self.insight_doc else ""
            }

            logger.info("\n" + "=" * 60)
            logger.info("Task completed")
            logger.info(f"Stats: {stats}")
            logger.info("=" * 60)

            # 显示完整Memory Bank记忆
            self._display_complete_memory()

            result = {
                "answer": answer or "Task finished but no explicit answer",
                "insight_doc": final_insight_doc,
                "stats": stats,
                "react_messages": last_react_result.get("messages", []) if last_react_result else []
            }

            return result

        except Exception as e:
            logger.error(f"Task execution failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

        finally:
            if self.insight_doc is not None:
                self.adapter.cleanup_temp_storage()
                self.query_graph.clear()
                self.retrieval_module.mark_index_dirty()
                self.interaction_tree.clear()
                self.insight_doc = None

    def _initialize(self, user_input: str) -> str:
        """
        初始化阶段

        流程：
        1. 解析用户输入
        2. 多模态临时存储
        3. 分类→结构化→检索→分析→建边
        4. 规划下一步
        5. 增强Prompt

        Args:
            user_input: 用户输入

        Returns:
            增强后的Prompt
        """
        # 1. 解析用户输入
        text_context, question = self._parse_user_input(user_input)
        doc_id = str(uuid.uuid4())

        # 2. 判断是否有文本上下文
        if not text_context:
            # 跳到规划
            self.insight_doc = InsightDoc(
                doc_id=doc_id,
                task_goal=question,
                completed_tasks=[],
                current_task=""
            )
            planning_output = self.planning_agent.run(PlanningInput(
                insight_doc=self.insight_doc
            ))
            self.insight_doc = InsightDoc(
                doc_id=doc_id,
                task_goal=planning_output.task_goal,
                completed_tasks=planning_output.completed_tasks,
                current_task=planning_output.current_task
            )
            return self.adapter.enhance_prompt(self.insight_doc)

        # 3. 分类/聚类
        classification_output = self.classification_agent.run(ClassificationInput(
            context=text_context,
            task_goal=question
        ))

        # 5-9. 对每个cluster进行处理
        new_nodes = []
        conflicts = []

        for i, cluster in enumerate(classification_output.clusters):

            # 5. 结构化
            structure_output = self.structure_agent.run(StructureInput(
                content=cluster.content,  # ✅ 修复：使用 cluster.content（原始文本）
                context=cluster.context,
                keywords=cluster.keywords,
                current_task=question  # 使用用户问题作为当前任务参考
            ))

            # 6. 组装节点
            node = self._create_node(
                summary=structure_output.summary,
                context=cluster.context,
                keywords=cluster.keywords  # 使用cluster的keywords，不是structure_output
            )
            self.graph_ops.add_node(node)
            new_nodes.append(node)

            # ✅ 优化：标记索引为脏（索引会在下次检索时自动重建）
            self.retrieval_module.mark_index_dirty()

            # 7. 检索相似节点
            candidates = self.retrieval_module.hybrid_retrieval(
                query_embedding=node.embedding,
                query_keywords=node.keywords,
                graph=self.query_graph,
                exclude_ids={node.id}
            )

            # 8. 分析关系
            if candidates:
                from src.agents.analysis_agent import AnalysisInput, NodeInfo
                analysis_input = AnalysisInput(
                    new_node=NodeInfo(
                        id=node.id,
                        summary=node.summary,
                        context=node.context,
                        keywords=node.keywords,
                        merge_description=node.merge_description
                    ),
                    candidate_nodes=[
                        NodeInfo(
                            id=c.id,
                            summary=c.summary,
                            context=c.context,
                            keywords=c.keywords,
                            merge_description=c.merge_description
                        )
                        for c in candidates
                    ]
                )
                analysis_output = self.analysis_agent.run(analysis_input)

                # 处理关系
                for rel in analysis_output.relationships:
                    if rel.relationship == "conflict":
                        conflicts.append({
                            "node1": node.id,
                            "node2": rel.existing_node_id,
                            "description": rel.conflict_description
                        })
                        logger.info(f"    ⚠️  Conflict detected: {rel.conflict_description[:150]}{'...' if len(rel.conflict_description or '') > 150 else ''}")
                    elif rel.relationship == "related":
                        self.graph_ops.add_edge(node.id, rel.existing_node_id)

            # 9. 创建Interaction Tree Entry（保存完整内容）
            self.interaction_tree.add_entry(node.id, cluster.content)

        # 10. 规划
        conflict_notification = None
        if conflicts:
            conflict_notification = ConflictNotification(
                conflicting_node_ids=[conflicts[0]["node1"], conflicts[0]["node2"]],
                conflict_description=conflicts[0]["description"]
            )
            # ✅ 修复：将冲突添加到adapter队列（与_handle_normal_task保持一致）
            # 确保执行循环中has_pending_conflicts()能正确识别冲突状态
            for conflict in conflicts:
                pair = [conflict["node1"], conflict["node2"]]
                if pair not in self.adapter._pending_conflicts:
                    self.adapter._pending_conflicts.append(pair)
            logger.info(f"  ⚠️  Conflict detected, cross-validation needed")

        # ✅ 修复：初始化insight_doc（在调用Planning Agent之前）
        # 确保Planning Agent接收到有效的InsightDoc对象，而不是None
        self.insight_doc = InsightDoc(
            doc_id=doc_id,
            task_goal=question,
            completed_tasks=[],
            current_task=""
        )

        print(f"\n[Planning] Calling Planning Agent - planning next task...")
        # ✅ 修复：传入当前的 insight_doc（包含已完成的任务），而不是空的 InsightDoc
        planning_output = self.planning_agent.run(PlanningInput(
            insight_doc=self.insight_doc,
            new_memory_nodes=[
                {
                    "id": node.id,
                    "context": node.context,
                    "keywords": node.keywords,
                    "summary": node.summary
                }
                for node in new_nodes
            ],
            conflict_notification=conflict_notification
        ))
        print(f"   [OK] Planning complete: current task={'yes' if planning_output.current_task else 'no'}")

        # ✅ 更新insight_doc（使用Planning Agent的输出）
        self.insight_doc.task_goal = planning_output.task_goal
        self.insight_doc.completed_tasks = planning_output.completed_tasks
        self.insight_doc.current_task = planning_output.current_task

        # 11. 增强Prompt
        return self.adapter.enhance_prompt(self.insight_doc)

    def _create_node(self, summary: str, context: str, keywords: List[str]) -> QueryGraphNode:
        """
        创建Query Graph节点

        Args:
            summary: 摘要
            context: 上下文
            keywords: 关键词列表

        Returns:
            QueryGraphNode实例
        """
        node_id = str(uuid.uuid4())
        timestamp = time.time()
        text = f"{summary} {context} {' '.join(keywords)}"
        embedding = self.embedding_module.compute_embedding(text)

        return QueryGraphNode(
            id=node_id,
            summary=summary,
            context=context,
            keywords=keywords,
            embedding=embedding,
            timestamp=timestamp,
            links=[]  # ✅ 修复：应该是list而不是set
        )

    def _parse_user_input(self, user_input: str) -> Tuple[str, str]:
        """
        解析用户输入

        支持格式：
        1. 纯问题
        2. "上下文：...\n问题：..."
        3. "Context:...\nQuestion:..."

        Args:
            user_input: 用户输入

        Returns:
            (text_context, question)
        """
        lines = user_input.split('\n')
        text_context = ""
        question = ""

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("上下文：") or line.startswith("Context:"):
                separator = "：" if "：" in line else ":"
                text_context = line.split(separator, 1)[1].strip()
            elif line.startswith("问题：") or line.startswith("Question:"):
                separator = "：" if "：" in line else ":"
                question = line.split(separator, 1)[1].strip()
            elif not question and not text_context:
                # 第一行且没有前缀，当作问题
                question = line

        # 如果没有明确问题，整个输入作为问题
        if not question:
            question = user_input

        return text_context, question

    def _extract_tool_responses(self, messages: List[Dict[str, str]]) -> List[str]:
        """
        从ReAct消息历史中提取工具响应内容（仅工具输出）

        Args:
            messages: ReAct消息历史

        Returns:
            工具响应文本列表
        """
        tool_responses = []

        for message in messages:
            if message.get("role") == "user" and "content" in message:
                content = message["content"]

                # 提取<tool_response>标签内容
                if '<tool_response>' in content and '</tool_response>' in content:
                    try:
                        response_text = content.split('<tool_response>')[1].split('</tool_response>')[0]
                        response_text = response_text.strip()

                        if response_text and response_text not in tool_responses:
                            tool_responses.append(response_text)
                    except Exception as e:
                        logger.warning(f"Failed to parse tool response: {str(e)}")

        return tool_responses

    def _extract_full_context(self, messages: List[Dict[str, str]]) -> str:
        """
        从ReAct消息历史中提取完整上下文（包括思考、工具调用、响应和答案）

        这个方法提取：
        1. ReAct Agent的完整响应（保留<think>、<tool_call>、<answer>等所有标签）
        2. 工具响应结果（保留<tool_response>标签）

        重要：保持原始标签格式，不做任何转换！

        Args:
            messages: ReAct消息历史

        Returns:
            完整的上下文字符串
        """
        context_parts = []

        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")

            if role == "assistant":
                # ✅ 修复：直接保留完整的assistant响应，不做任何转换
                # 包含<think>、<tool_call>、<answer>等所有标签
                if content.strip():
                    context_parts.append(content.strip())

            elif role == "user":
                # 提取工具响应（保留<tool_response>标签）
                if '<tool_response>' in content and '</tool_response>' in content:
                    context_parts.append(content.strip())

        full_context = "\n\n".join(context_parts)

        return full_context

    def _should_terminate(self) -> bool:
        """
        判断是否终止

        Returns:
            是否应该终止任务
        """
        if not self.insight_doc:
            return False

        # 如果没有待办任务且所有已完成任务都成功，则终止
        no_pending = not self.insight_doc.current_task
        all_success = all(
            task.status == "成功"
            for task in self.insight_doc.completed_tasks
        ) if self.insight_doc.completed_tasks else True

        return no_pending and all_success

    def _display_complete_memory(self):
        """显示完整的Memory Bank记忆"""

        # 1. Query Graph展示
        logger.info("\n" + "=" * 80)
        logger.info("📊 Query Graph - Semantic Memory Graph")
        logger.info("=" * 80)
        logger.info(f"Total nodes: {self.query_graph.get_node_count()}")
        logger.info(f"Total edges: {self.query_graph.get_edge_count()}")
        logger.info("")

        for i, node in enumerate(self.query_graph.get_all_nodes(), 1):
            logger.info(f"Node {i}:")
            logger.info(f"  ID: {node.id}")
            logger.info(f"  Topic: {node.context}")
            logger.info(f"  Keywords: {', '.join(node.keywords)}")
            logger.info(f"  Summary: {node.summary[:200]}{'...' if len(node.summary) > 200 else ''}")
            logger.info(f"  Neighbors: {len(node.links)}")
            if node.links:
                logger.info(f"  Linked node IDs: {', '.join([nid[:8] for nid in node.links[:3]])}{'...' if len(node.links) > 3 else ''}")
            logger.info("-" * 80)

        # 2. Interaction Tree展示
        logger.info("\n" + "=" * 80)
        logger.info("📚 Interaction Tree - Interaction History")
        logger.info("=" * 80)
        logger.info(f"Total entries: {self.interaction_tree.get_total_entries()}")
        logger.info(f"Linked nodes: {len(self.interaction_tree.get_nodes_with_entries())}")
        logger.info("")

        for node_id in self.interaction_tree.get_nodes_with_entries():
            text = self.interaction_tree.get_entry(node_id)
            logger.info(f"Node ID: {node_id[:8]}...")
            if text:
                logger.info(f"  Text: {text[:150]}{'...' if len(text) > 150 else ''}")
            logger.info("-" * 80)

        # 3. Insight Doc展示
        if self.insight_doc:
            logger.info("\n" + "=" * 80)
            logger.info("📝 Insight Doc - Task Status")
            logger.info("=" * 80)
            logger.info(f"Task goal: {self.insight_doc.task_goal}")
            logger.info(f"Completed tasks: {len(self.insight_doc.completed_tasks)}")
            logger.info("")

            for i, task in enumerate(self.insight_doc.completed_tasks, 1):
                logger.info(f"Task {i}:")
                logger.info(f"  Type: {task.type.value}")
                logger.info(f"  Description: {task.description}")
                logger.info(f"  Status: {task.status}")
                logger.info(f"  Context: {task.context}")
                logger.info("-" * 40)

            logger.info(f"Current task: {self.insight_doc.current_task if self.insight_doc.current_task else '(none)'}")
            logger.info("=" * 80)

    def export_memory(self, filepath: str):
        """
        导出记忆到JSON文件

        Args:
            filepath: 输出文件路径
        """
        import json

        memory_data = {
            "insight_doc": self.insight_doc.to_dict() if self.insight_doc else None,
            "query_graph": self.query_graph.to_dict(),
            "interaction_tree": self.interaction_tree.to_dict()
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Memory exported to: {filepath}")

    def load_memory(self, filepath: str):
        """
        从JSON文件加载记忆

        Args:
            filepath: 输入文件路径
        """
        import json

        with open(filepath, 'r', encoding='utf-8') as f:
            memory_data = json.load(f)

        if memory_data.get("insight_doc"):
            self.insight_doc = InsightDoc.from_dict(memory_data["insight_doc"])

        self.query_graph = QueryGraph.from_dict(memory_data["query_graph"])
        self.interaction_tree = InteractionTree.from_dict(memory_data["interaction_tree"])

        logger.info(f"Memory loaded from file: {filepath}")

    def __repr__(self) -> str:
        """返回Memory Bank摘要"""
        return (
            f"MemoryBank("
            f"nodes={self.query_graph.get_node_count()}, "
            f"edges={self.query_graph.get_edge_count()}, "
            f"entries={self.interaction_tree.get_total_entries()})"
        )
