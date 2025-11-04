"""
Agentic Memory Bank 核心类

整合所有组件，提供完整的记忆管理功能。

规范文档：第8章
"""

import logging
import uuid
import time
from typing import Dict, Any, List, Optional, Tuple

from src.storage.insight_doc import InsightDoc, CompletedTask, TaskType
from src.storage.query_graph import QueryGraph, QueryGraphNode
from src.storage.interaction_tree import InteractionTree, create_entry
from src.modules.embedding import EmbeddingModule
from src.modules.retrieval import RetrievalModule
from src.modules.graph_ops import GraphOperations
from src.modules.context_update import ContextUpdateModule
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
        logger.info("初始化Agentic Memory Bank...")

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
        self.context_updater = ContextUpdateModule(self.query_graph, self.embedding_module)

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

        # Visit工具：如果配置了Jina API key，使用Jina Reader；否则使用BeautifulSoup
        jina_api_key = self.config.JINA_API_KEY
        self.visit_tool = VisitTool(
            llm_client=self.llm_client,
            jina_api_key=jina_api_key
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
            max_context_tokens=self.config.MAX_CONTEXT_TOKENS
        )

        logger.info("Agentic Memory Bank初始化完成")

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
        logger.info(f"开始新任务: {user_input[:100]}...")
        logger.info("=" * 60)

        try:
            # 1. 初始化阶段
            print("\n" + "🚀 " + "=" * 68)
            print("  Agentic Memory Bank - 初始化")
            print("=" * 70)
            enhanced_prompt = self._initialize(user_input)
            iterations = 0

            # 显示任务目标和初始状态
            if self.insight_doc:
                print(f"\n📋 任务目标: {self.insight_doc.task_goal}")
                print(f"📝 待办任务: {self.insight_doc.pending_tasks}")
                print("=" * 70)

            # 2. 执行循环
            answer = None
            last_react_result = None  # 保存最后一次ReAct结果
            max_iterations = self.config.MAX_LLM_CALL_PER_RUN
            while not self._should_terminate() and iterations < max_iterations:
                iterations += 1
                print(f"\n{'🔄 ' + '=' * 68}")
                print(f"  执行轮次 {iterations}")
                print("=" * 70)
                logger.info(f"\n----- 执行轮次 {iterations} -----")

                # 显示当前任务状态
                if self.insight_doc:
                    if self.insight_doc.pending_tasks:
                        print(f"⏳ 当前任务: {self.insight_doc.pending_tasks[0]}")
                    print(f"✅ 已完成: {len(self.insight_doc.completed_tasks)} 个任务")
                    print(f"📊 记忆节点: {self.query_graph.get_node_count()} 个")

                # 2.1 ReAct执行
                react_result = self.react_agent.run(enhanced_prompt)
                last_react_result = react_result  # 保存结果
                answer = react_result.get("prediction", "")

                logger.info(f"ReAct终止原因: {react_result.get('termination')}")

                # 2.2 上下文拦截：提取搜索结果并转化为记忆
                # 检查是否是最终回答任务
                current_task = self.insight_doc.pending_tasks[0] if self.insight_doc and self.insight_doc.pending_tasks else ""
                is_final_answer_task = "根据现有相关记忆直接回答问题" in current_task or "根据现有记忆回答问题" in current_task

                if self.insight_doc and self.insight_doc.pending_tasks:
                    # 检查是否有工具调用（通过检查消息历史）
                    messages = react_result.get("messages", [])
                    has_tool_calls = any(
                        msg.get("role") == "assistant" and "<tool_call>" in msg.get("content", "")
                        for msg in messages
                    )

                    tool_responses = self._extract_tool_responses(messages)

                    if has_tool_calls and not is_final_answer_task:
                        # 有工具调用，且不是最终回答任务 - 进入记忆处理流程
                        print(f"\n🧠 开始整理记忆...")
                        logger.info(f"开始整理记忆：提取完整上下文并转化为记忆节点")

                        # 提取完整上下文（包括思考过程、工具调用和工具响应）
                        full_context = self._extract_full_context(messages)
                        logger.debug(f"完整上下文长度: {len(full_context)} 字符")

                        # 调用上下文拦截机制
                        try:
                            # 判断任务类型（参考 REQUIREMENTS_FINAL.md 第4.2节）
                            if "验证" in current_task or "Cross Validation" in current_task or "交叉验证" in current_task:
                                task_type = "CROSS_VALIDATE"
                            else:
                                task_type = "NORMAL"

                            logger.info(f"任务类型: {task_type}, 当前任务: {current_task}")

                            # intercept_context 内部会调用 Planning Agent 更新 insight_doc
                            # 包括 completed_tasks 和 pending_tasks
                            self.adapter.intercept_context(full_context, task_type, self.insight_doc)

                            print(f"✅ 记忆处理完成")
                            logger.info(f"记忆处理完成，待办任务={len(self.insight_doc.pending_tasks)}")

                        except Exception as e:
                            print(f"❌ 错误: {str(e)}")
                            logger.error(f"上下文拦截失败: {str(e)}")
                            import traceback
                            traceback.print_exc()

                    elif is_final_answer_task and react_result.get("termination") == "answer":
                        # ✅ 修复：最终回答任务且有答案 - 直接保存答案并结束
                        answer = react_result.get("prediction", "")  # ← 从prediction字段获取
                        print("\n✅ 获得最终答案，任务完成！")
                        logger.info(f"最终答案任务完成: {answer[:100]}...")

                        # 标记任务完成，清空pending_tasks
                        self.insight_doc.pending_tasks = []
                        completed_task = CompletedTask(
                            type=TaskType.NORMAL,
                            description=current_task,
                            status="成功",
                            context=f"最终答案: {answer[:100]}"
                        )
                        self.insight_doc.completed_tasks.append(completed_task)
                        break  # 直接结束循环

                    elif not tool_responses:
                        # 没有工具调用，但ReAct返回了答案 - 直接标记任务完成
                        print("\n💡 ReAct直接回答了问题（未调用工具）")
                        logger.info("ReAct直接回答，未调用工具")

                        if self.insight_doc.pending_tasks:
                            # 保存答案到外层变量（从prediction字段获取）
                            answer = react_result.get("prediction", "")

                            # 标记任务完成
                            self.insight_doc.pending_tasks.remove(current_task)
                            completed_task = CompletedTask(
                                type=TaskType.NORMAL,
                                description=current_task,
                                status="成功",
                                context=f"直接回答: {answer[:200]}"
                            )
                            self.insight_doc.completed_tasks.append(completed_task)
                            print(f"✅ 任务完成: {current_task}")
                            logger.info(f"任务已完成: {current_task}")

                            # 如果是最终回答任务，直接结束循环
                            if is_final_answer_task:
                                print("\n✅ 获得最终答案，任务完成！")
                                logger.info(f"最终答案: {answer[:100]}...")
                                break

                            # 调用Planning Agent检查是否还有其他任务
                            from src.agents.planning_agent import PlanningInput
                            planning_output = self.planning_agent.run(PlanningInput(
                                insight_doc=self.insight_doc,
                                new_memory_nodes=[],  # 没有生成记忆节点
                                conflict_notification=None
                            ))

                            # 更新insight_doc
                            self.insight_doc.pending_tasks = planning_output.pending_tasks
                            logger.info(f"Planning Agent更新: 待办任务={len(self.insight_doc.pending_tasks)}")

                # 2.3 检查是否有待办任务（如果没有，说明任务完成）
                if not self.insight_doc or not self.insight_doc.pending_tasks:
                    print("\n🎉 所有任务完成！")
                    logger.info("任务完成（无待办任务）")
                    break

                # 2.4 如果ReAct已经给出答案且无更多待办任务，结束
                if react_result.get("termination") == "answer" and not self.insight_doc.pending_tasks:
                    print("\n🎉 ReAct Agent已给出答案，任务完成！")
                    break

                # 2.5 增强下一轮Prompt（基于新的任务状态）
                print(f"\n🔄 准备下一轮执行...")
                if self.insight_doc.pending_tasks:
                    print(f"📋 下一步任务: {self.insight_doc.pending_tasks[0]}")
                enhanced_prompt = self.adapter.enhance_prompt(self.insight_doc)

            # 3. 统计信息
            final_insight_doc = self.insight_doc.to_dict() if self.insight_doc else None

            stats = {
                "iterations": iterations,
                "graph_nodes": self.query_graph.get_node_count(),
                "graph_edges": self.query_graph.get_edge_count(),
                "tree_entries": self.interaction_tree.get_total_entries(),
                "completed_tasks": len(self.insight_doc.completed_tasks) if self.insight_doc else 0,
                "pending_tasks": len(self.insight_doc.pending_tasks) if self.insight_doc else 0
            }

            logger.info("\n" + "=" * 60)
            logger.info("Task completed")
            logger.info(f"Stats: {stats}")
            logger.info("=" * 60)

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
        logger.info("\n----- 初始化阶段 -----")

        # 1. 解析用户输入
        text_context, question = self._parse_user_input(user_input)
        doc_id = str(uuid.uuid4())

        logger.info(f"解析结果: 上下文长度={len(text_context)}, 问题={question[:50]}...")

        # 2. 判断是否有文本上下文
        if not text_context:
            logger.info("无文本上下文，直接进入规划")
            # 跳到规划
            self.insight_doc = InsightDoc(
                doc_id=doc_id,
                task_goal=question,
                completed_tasks=[],
                pending_tasks=[]
            )
            planning_output = self.planning_agent.run(PlanningInput(
                insight_doc=self.insight_doc
            ))
            self.insight_doc = InsightDoc(
                doc_id=doc_id,
                task_goal=planning_output.task_goal,
                completed_tasks=planning_output.completed_tasks,
                pending_tasks=planning_output.pending_tasks
            )
            return self.adapter.enhance_prompt(self.insight_doc)

        # 3. 分类/聚类
        logger.info("执行分类/聚类...")
        classification_output = self.classification_agent.run(ClassificationInput(
            context=text_context,
            task_goal=question
        ))
        logger.info(f"分类结果: should_cluster={classification_output.should_cluster}, "
                   f"clusters={len(classification_output.clusters)}")

        # 5-9. 对每个cluster进行处理
        new_nodes = []
        conflicts = []

        for i, cluster in enumerate(classification_output.clusters):
            logger.info(f"\n处理Cluster {i+1}/{len(classification_output.clusters)}: {cluster.context}")

            # 5. 结构化
            structure_output = self.structure_agent.run(StructureInput(
                content=cluster.content,  # ✅ 修复：使用 cluster.content（原始文本）
                context=cluster.context,
                keywords=cluster.keywords
            ))
            logger.debug(f"  结构化完成: summary长度={len(structure_output.summary)}")

            # 6. 组装节点
            node = self._create_node(
                summary=structure_output.summary,
                context=cluster.context,
                keywords=cluster.keywords  # 使用cluster的keywords，不是structure_output
            )
            self.graph_ops.add_node(node)
            new_nodes.append(node)
            logger.debug(f"  新节点已添加: {node.id[:8]}...")

            # ✅ 优化：标记索引为脏（索引会在下次检索时自动重建）
            self.retrieval_module.mark_index_dirty()

            # 7. 检索相似节点
            candidates = self.retrieval_module.hybrid_retrieval(
                query_embedding=node.embedding,
                query_keywords=node.keywords,
                graph=self.query_graph,
                exclude_ids={node.id}
            )
            logger.debug(f"  检索到 {len(candidates)} 个候选节点")

            # 8. 分析关系
            if candidates:
                from src.agents.analysis_agent import AnalysisInput, NodeInfo
                analysis_input = AnalysisInput(
                    new_node=NodeInfo(
                        id=node.id,
                        summary=node.summary,
                        context=node.context,
                        keywords=node.keywords
                    ),
                    candidate_nodes=[
                        NodeInfo(
                            id=c.id,
                            summary=c.summary,
                            context=c.context,
                            keywords=c.keywords
                        )
                        for c in candidates
                    ]
                )
                analysis_output = self.analysis_agent.run(analysis_input)
                logger.debug(f"  分析完成: {len(analysis_output.relationships)} 个关系")

                # 处理关系
                for rel in analysis_output.relationships:
                    if rel.relationship == "conflict":
                        conflicts.append({
                            "node1": node.id,
                            "node2": rel.existing_node_id,
                            "description": rel.conflict_description
                        })
                        logger.info(f"    ⚠️  检测到冲突: {rel.conflict_description[:50]}...")
                    elif rel.relationship == "related":
                        self.graph_ops.add_edge(node.id, rel.existing_node_id)
                        logger.debug(f"    建立关联边: {node.id[:8]}... <-> {rel.existing_node_id[:8]}...")

                        # 更新上下文
                        if rel.context_update_new:
                            self.context_updater.update_node_context(
                                node_id=node.id,
                                new_context=rel.context_update_new,
                                new_keywords=rel.keywords_update_new
                            )
                        if rel.context_update_existing:
                            self.context_updater.update_node_context(
                                node_id=rel.existing_node_id,
                                new_context=rel.context_update_existing,
                                new_keywords=rel.keywords_update_existing
                            )

            # 9. 创建Interaction Tree Entry
            entry = create_entry(
                text=cluster.content,  # ✅ 修复：保存完整内容而不是一句话摘要
                metadata={"source": "user_input", "cluster_id": cluster.cluster_id}
            )
            self.interaction_tree.add_entry(node.id, entry)
            logger.debug(f"  Interaction Tree Entry已创建")

        # 10. 规划
        logger.debug("执行任务规划...")
        conflict_notification = None
        if conflicts:
            conflict_notification = ConflictNotification(
                conflicting_node_ids=[conflicts[0]["node1"], conflicts[0]["node2"]],
                conflict_description=conflicts[0]["description"]
            )
            logger.info(f"  ⚠️  检测到冲突，需要交叉验证")

        print(f"\n📅 调用 Planning Agent - 规划下一步任务...")
        planning_output = self.planning_agent.run(PlanningInput(
            insight_doc=InsightDoc(
                doc_id=doc_id,
                task_goal=question,
                completed_tasks=[],
                pending_tasks=[]
            ),
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
        print(f"   ✅ 规划完成: {len(planning_output.pending_tasks)} 个待办任务")

        self.insight_doc = InsightDoc(
            doc_id=doc_id,
            task_goal=planning_output.task_goal,
            completed_tasks=planning_output.completed_tasks,
            pending_tasks=planning_output.pending_tasks
        )

        logger.info(f"规划完成: 待办任务={len(self.insight_doc.pending_tasks)}")

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
                            logger.debug(f"提取工具响应: {response_text[:100]}...")
                    except Exception as e:
                        logger.warning(f"解析工具响应失败: {str(e)}")

        return tool_responses

    def _extract_full_context(self, messages: List[Dict[str, str]]) -> str:
        """
        从ReAct消息历史中提取完整上下文（包括思考、工具调用和响应）

        这个方法提取：
        1. ReAct Agent的思考过程（<think>标签）
        2. 工具调用和参数
        3. 工具响应结果

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
                # 提取思考过程
                if '<think>' in content and '</think>' in content:
                    try:
                        think_text = content.split('<think>')[1].split('</think>')[0].strip()
                        if think_text:
                            context_parts.append(f"【分析过程】\n{think_text}")
                    except Exception as e:
                        logger.warning(f"解析思考内容失败: {str(e)}")

                # ✅ 修复：提取工具调用参数（包含重要语义信息）
                if '<tool_call>' in content and '</tool_call>' in content:
                    try:
                        tool_call_text = content.split('<tool_call>')[1].split('</tool_call>')[0].strip()
                        if tool_call_text:
                            # 解析JSON并格式化
                            import json
                            tool_call_json = json.loads(tool_call_text)
                            tool_name = tool_call_json.get('name', '')
                            tool_args = tool_call_json.get('arguments', {})
                            context_parts.append(f"【工具调用】\n工具: {tool_name}\n参数: {json.dumps(tool_args, ensure_ascii=False)}")
                    except Exception as e:
                        logger.warning(f"解析工具调用失败: {str(e)}")

            elif role == "user":
                # 提取工具响应
                if '<tool_response>' in content and '</tool_response>' in content:
                    try:
                        response_text = content.split('<tool_response>')[1].split('</tool_response>')[0].strip()
                        if response_text:
                            context_parts.append(f"【工具输出】\n{response_text}")
                    except Exception as e:
                        logger.warning(f"解析工具响应失败: {str(e)}")

        full_context = "\n\n".join(context_parts)
        logger.debug(f"提取完整上下文: {len(context_parts)} 个部分，总长度 {len(full_context)} 字符")

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
        no_pending = len(self.insight_doc.pending_tasks) == 0
        all_success = all(
            task.status == "成功"
            for task in self.insight_doc.completed_tasks
        ) if self.insight_doc.completed_tasks else True

        return no_pending and all_success

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

        logger.info(f"记忆已导出到: {filepath}")

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

        logger.info(f"记忆已从文件加载: {filepath}")

    def __repr__(self) -> str:
        """返回Memory Bank摘要"""
        return (
            f"MemoryBank("
            f"nodes={self.query_graph.get_node_count()}, "
            f"edges={self.query_graph.get_edge_count()}, "
            f"entries={self.interaction_tree.get_total_entries()})"
        )
