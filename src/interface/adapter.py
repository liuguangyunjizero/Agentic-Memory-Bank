"""
MemoryBankAdapter

Adapter layer bridging the MemoryBank core with the host agent framework.
It is responsible for prompt enhancement, context interception, and conflict routing.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Dict, List, Optional

from src.agents.analysis_agent import AnalysisInput, NodeInfo
from src.agents.classification_agent import ClassificationInput
from src.agents.integration_agent import IntegrationInput, NodeWithNeighbors
from src.agents.planning_agent import ConflictNotification, PlanningInput
from src.agents.structure_agent import StructureInput
from src.storage.insight_doc import TaskType
from src.storage.interaction_tree import MergeEvent, create_entry
from src.storage.query_graph import QueryGraphNode

logger = logging.getLogger(__name__)


class MemoryBankAdapter:
    """Bridge between MemoryBank and the external execution loop."""

    def __init__(self, memory_bank, retrieval_module, file_utils) -> None:
        self.memory_bank = memory_bank
        self.retrieval = retrieval_module
        self.file_utils = file_utils
        self._pending_conflicts: List[List[str]] = []
        logger.info("MemoryBankAdapter initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cleanup_temp_storage(self) -> None:
        """Cleanup temporary files and reset conflict queue."""
        try:
            self.file_utils.cleanup_temp_dir()
        except Exception as exc:  # pragma: no cover - best effort cleanup
            logger.warning("Failed to cleanup temp dir: %s", exc)
        self._pending_conflicts.clear()

    def enhance_prompt(self, insight_doc) -> str:
        """Return the <task>/<memory> enhanced prompt for ReAct."""
        if insight_doc is None:
            return "<task>\nNo active task\n</task>\n\n<memory>\nNo related memory\n</memory>"

        task_section = self._build_task_section(insight_doc)
        memory_section = self._build_memory_section(insight_doc)

        prompt = (
            f"<task>\n{task_section}\n</task>\n\n"
            f"<memory>\n{memory_section}\n</memory>\n\n"
            "Please continue with the next step."
        )
        logger.debug("Enhanced prompt length=%d", len(prompt))
        return prompt

    def intercept_context(self, context: str, task_type: str, insight_doc) -> None:
        """Convert tool output into structured memory or resolve conflicts."""
        logger.info("Intercepting context: type=%s, length=%d", task_type, len(context))
        if task_type == "CROSS_VALIDATE":
            self._handle_conflict_resolution(context, insight_doc)
        else:
            self._handle_normal_task(context, insight_doc)

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def _build_task_section(self, insight_doc) -> str:
        lines: List[str] = [f"Task goal: {insight_doc.task_goal}"]

        if insight_doc.completed_tasks:
            lines.append("")
            lines.append("Completed tasks:")
            for idx, task in enumerate(insight_doc.completed_tasks, 1):
                lines.append(f"{idx}. [{task.type.value}] {task.description} - {task.status}")
                lines.append(f"   Context: {task.context}")

        if insight_doc.current_task:
            lines.append("")
            lines.append("Current task:")
            lines.append(f"1. {insight_doc.current_task}")
        else:
            lines.append("")
            lines.append("Current task: none")

        return "\n".join(lines)

    def _build_memory_section(self, insight_doc) -> str:
        if not insight_doc.current_task:
            return "No related memory"

        query_text = insight_doc.current_task
        query_embedding = self.memory_bank.embedding_module.compute_embedding(query_text)

        # 提取关键词：使用正则提取中文词组（2+字符）和英文单词（2+字符）
        import re
        words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]{2,}', query_text)
        query_keywords = words[:5] if words else [query_text]

        relevant_nodes = self.retrieval.hybrid_retrieval(
            query_embedding=query_embedding,
            query_keywords=query_keywords,
            graph=self.memory_bank.query_graph,
            exclude_ids=set(),
        )

        if not relevant_nodes:
            return "No related memory"

        lines: List[str] = []
        for idx, node in enumerate(relevant_nodes, 1):
            lines.append(f"Memory {idx}:")
            lines.append(f"Node ID: {node.id}")
            lines.append(f"Topic: {node.context}")
            lines.append(f"Keywords: {', '.join(node.keywords)}")
            lines.append(f"Summary: {node.summary}")
            lines.append("")

        return "\n".join(lines).strip()

    # ------------------------------------------------------------------
    # Conflict resolution
    # ------------------------------------------------------------------

    def _handle_conflict_resolution(self, validation_result: str, insight_doc) -> None:
        conflict_ids = self._get_conflicting_node_ids()
        if not conflict_ids:
            logger.warning("No pending conflicts found; skipping resolution.")
            return

        nodes_to_merge: List[NodeWithNeighbors] = []
        for node_id in conflict_ids:
            node = self.memory_bank.query_graph.get_node(node_id)
            if not node:
                logger.warning("Conflict node missing: %s", node_id)
                continue

            neighbours = self.memory_bank.graph_ops.get_neighbors(node_id)
            nodes_to_merge.append(
                NodeWithNeighbors(
                    id=node.id,
                    summary=node.summary,
                    context=node.context,
                    keywords=node.keywords,
                    neighbors=[
                        {"id": n.id, "context": n.context, "keywords": n.keywords}
                        for n in neighbours
                    ],
                )
            )

        if not nodes_to_merge:
            logger.error("Unable to load conflict nodes; aborting integration.")
            return

        integration_output = self.memory_bank.integration_agent.run(
            IntegrationInput(nodes_to_merge=nodes_to_merge, validation_result=validation_result)
        )

        merged = integration_output.merged_node
        embedding_text = (
            f"{merged['summary']} {merged['context']} {' '.join(merged['keywords'])}"
        )
        new_node = QueryGraphNode(
            id=str(uuid.uuid4()),
            summary=merged["summary"],
            context=merged["context"],
            keywords=merged["keywords"],
            embedding=self.memory_bank.embedding_module.compute_embedding(embedding_text),
            timestamp=time.time(),
            links=[],
        )

        self.memory_bank.graph_ops.merge_nodes(conflict_ids, new_node)
        self.retrieval.mark_index_dirty()

        if integration_output.neighbor_updates:
            updates: Dict[str, Dict[str, Optional[List[str]]]] = {}
            for neighbour_id, payload in integration_output.neighbor_updates.items():
                updates[neighbour_id] = {
                    "context": payload.get("context"),
                    "keywords": payload.get("keywords"),
                }
            self.memory_bank.context_updater.batch_update_node_contexts(updates)
            self.retrieval.mark_index_dirty()

        merge_event = MergeEvent(
            event_id=str(uuid.uuid4()),
            merged_node_ids=conflict_ids,
            new_node_id=new_node.id,
            timestamp=time.time(),
            description=integration_output.interaction_tree_description,
        )
        self.memory_bank.interaction_tree.record_merge(merge_event)

        # 为cross_validate操作创建interaction tree entry
        from ..storage.interaction_tree import create_entry
        validate_entry = create_entry(
            text=validation_result,
            metadata={"source": "cross_validate", "merged_node_ids": conflict_ids},
        )
        self.memory_bank.interaction_tree.add_entry(new_node.id, validate_entry)

        if insight_doc.current_task:
            pending_desc = insight_doc.current_task
            # 从merged_node的summary中提取核心信息作为context
            # summary应该包含了Integration Agent整合后的核心结论
            context_summary = merged["summary"][:200] if len(merged["summary"]) > 200 else merged["summary"]
            validation_context = f"{context_summary}"

            insight_doc.add_completed_task(
                task_type=TaskType.CROSS_VALIDATE,
                description=pending_desc,
                status="成功",
                context=validation_context,
            )

        planning_output = self.memory_bank.planning_agent.run(
            PlanningInput(
                insight_doc=insight_doc,
                new_memory_nodes=[],
                conflict_notification=None,
            )
        )
        insight_doc.task_goal = planning_output.task_goal
        insight_doc.completed_tasks = planning_output.completed_tasks
        insight_doc.current_task = planning_output.current_task

    # ------------------------------------------------------------------
    # Normal task handling
    # ------------------------------------------------------------------

    def _handle_normal_task(self, context: str, insight_doc) -> None:
        # 提取当前任务（如果没有任务，使用 task_goal）
        current_task = insight_doc.current_task if insight_doc.current_task else insight_doc.task_goal

        logger.info("🔍 步骤1/4: 调用 Classification Agent 解析上下文...")

        # Classification Agent 输入
        logger.debug("="*80)
        logger.debug("📝 Classification Agent 输入:")
        logger.debug(f"  Current Task: {current_task}")
        logger.debug(f"  Context 长度: {len(context)} 字符")
        logger.debug("="*80)

        classification_output = self.memory_bank.classification_agent.run(
            ClassificationInput(
                context=context,
                task_goal=insight_doc.task_goal,
                current_task=current_task
            )
        )

        # Classification Agent 输出
        logger.debug("="*80)
        logger.debug("📤 Classification Agent 输出:")
        logger.debug(f"  是否需要分类: {classification_output.should_cluster}")
        logger.debug(f"  Cluster 数量: {len(classification_output.clusters)}")
        for idx, cluster in enumerate(classification_output.clusters, 1):
            logger.debug(f"  Cluster {idx}: {cluster.context} (关键词: {', '.join(cluster.keywords)})")
        logger.debug("="*80)

        new_nodes: List[QueryGraphNode] = []
        conflicts: List[Dict[str, str]] = []
        context_updates: Dict[str, Dict[str, Optional[List[str]]]] = {}

        logger.info(f"📝 步骤2/4: 处理 {len(classification_output.clusters)} 个 cluster（调用 Structure + Analysis Agent）...")
        for cluster in classification_output.clusters:
            logger.info(f"  - 调用 Structure Agent 压缩内容...")

            # Structure Agent 输入（精简版：不记录完整content）
            logger.debug("="*80)
            logger.debug("📝 Structure Agent 输入:")
            logger.debug(f"  Current Task: {current_task}")
            logger.debug(f"  Context: {cluster.context}")
            logger.debug(f"  Keywords: {cluster.keywords}")
            logger.debug(f"  Content 长度: {len(cluster.content)} 字符")
            logger.debug("="*80)

            structure_output = self.memory_bank.structure_agent.run(
                StructureInput(
                    content=cluster.content,
                    context=cluster.context,
                    keywords=cluster.keywords,
                    current_task=current_task,
                )
            )

            # Structure Agent 输出
            logger.debug("="*80)
            logger.debug("📤 Structure Agent 输出:")
            logger.debug(structure_output.summary)
            logger.debug("="*80)

            logger.info(f"  - 创建新节点（计算 embedding 用于检索）...")
            node = self.memory_bank._create_node(
                summary=structure_output.summary,
                context=cluster.context,
                keywords=cluster.keywords,
            )
            self.memory_bank.graph_ops.add_node(node)
            new_nodes.append(node)
            self.retrieval.mark_index_dirty()

            logger.info(f"  - 检索相似节点...")
            candidates = self.retrieval.hybrid_retrieval(
                query_embedding=node.embedding,
                query_keywords=node.keywords,
                graph=self.memory_bank.query_graph,
                exclude_ids={node.id},
            )

            if candidates:
                logger.info(f"  - 找到 {len(candidates)} 个候选节点，调用 Analysis Agent 分析关系...")

                # Analysis Agent 输入
                logger.debug("="*80)
                logger.debug("📝 Analysis Agent 输入:")
                logger.debug(f"  新节点: {node.context}")
                logger.debug(f"  候选节点数量: {len(candidates)}")
                logger.debug("="*80)

                analysis_output = self.memory_bank.analysis_agent.run(
                    AnalysisInput(
                        new_node=NodeInfo(
                            id=node.id,
                            summary=node.summary,
                            context=node.context,
                            keywords=node.keywords,
                        ),
                        candidate_nodes=[
                            NodeInfo(
                                id=c.id,
                                summary=c.summary,
                                context=c.context,
                                keywords=c.keywords,
                            )
                            for c in candidates
                        ],
                    )
                )

                # Analysis Agent 输出
                logger.debug("="*80)
                logger.debug("📤 Analysis Agent 输出:")
                logger.debug(f"  关系数量: {len(analysis_output.relationships)}")
                for rel in analysis_output.relationships:
                    logger.debug(f"  - {rel.relationship}: {rel.reasoning[:100]}...")
                logger.debug("="*80)

                conflict_rels = [
                    rel for rel in analysis_output.relationships if rel.relationship == "conflict"
                ]
                if conflict_rels:
                    for rel in conflict_rels:
                        conflicts.append(
                            {
                                "new": node.id,
                                "existing": rel.existing_node_id,
                                "description": rel.conflict_description or "冲突",
                            }
                        )
                    continue  # skip related updates when conflict exists

                for rel in analysis_output.relationships:
                    if rel.relationship != "related":
                        continue

                    if (
                        self.memory_bank.query_graph.has_node(node.id)
                        and self.memory_bank.query_graph.has_node(rel.existing_node_id)
                    ):
                        self.memory_bank.graph_ops.add_edge(node.id, rel.existing_node_id)

                    if rel.context_update_new:
                        context_updates[node.id] = {
                            "context": rel.context_update_new,
                            "keywords": rel.keywords_update_new or node.keywords,
                        }
                    if rel.context_update_existing:
                        existing_node = self.memory_bank.query_graph.get_node(rel.existing_node_id)
                        fallback_keywords = existing_node.keywords if existing_node else []
                        context_updates[rel.existing_node_id] = {
                            "context": rel.context_update_existing,
                            "keywords": rel.keywords_update_existing or fallback_keywords,
                        }

            entry = create_entry(
                text=cluster.content,
                metadata={"source": "react_tool", "cluster_id": cluster.cluster_id},
            )
            self.memory_bank.interaction_tree.add_entry(node.id, entry)

        if context_updates:
            logger.info(f"🔗 步骤3/4: 批量更新 {len(context_updates)} 个已有节点（批量计算 embedding）...")
            self.memory_bank.context_updater.batch_update_node_contexts(context_updates)
            self.retrieval.mark_index_dirty()

        conflict_notification = None
        if conflicts:
            first = conflicts[0]
            conflict_notification = ConflictNotification(
                conflicting_node_ids=[first["new"], first["existing"]],
                conflict_description=first["description"],
            )
            for conflict in conflicts:
                pair = [conflict["new"], conflict["existing"]]
                if pair not in self._pending_conflicts:
                    self._pending_conflicts.append(pair)

        logger.info("🎯 步骤4/4: 调用 Planning Agent 规划下一步任务...")

        # Planning Agent 输入
        logger.debug("="*80)
        logger.debug("📝 Planning Agent 输入:")
        logger.debug(f"  Task Goal: {insight_doc.task_goal}")
        logger.debug(f"  已完成任务: {len(insight_doc.completed_tasks)}")
        logger.debug(f"  当前任务: {insight_doc.current_task}")
        logger.debug(f"  新记忆节点: {len(new_nodes)}")
        logger.debug(f"  冲突通知: {'是' if conflict_notification else '否'}")
        logger.debug("="*80)

        planning_output = self.memory_bank.planning_agent.run(
            PlanningInput(
                insight_doc=insight_doc,
                new_memory_nodes=[
                    {
                        "id": node.id,
                        "context": node.context,
                        "keywords": node.keywords,
                        "summary": node.summary,
                    }
                    for node in new_nodes
                ],
                conflict_notification=conflict_notification,
            )
        )

        # Planning Agent 输出
        logger.debug("="*80)
        logger.debug("📤 Planning Agent 输出:")
        logger.debug(f"  Task Goal: {planning_output.task_goal}")
        logger.debug(f"  已完成任务: {len(planning_output.completed_tasks)}")
        logger.debug(f"  当前任务: {planning_output.current_task}")
        logger.debug("="*80)

        insight_doc.task_goal = planning_output.task_goal
        insight_doc.completed_tasks = planning_output.completed_tasks
        insight_doc.current_task = planning_output.current_task

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_conflicting_node_ids(self) -> Optional[List[str]]:
        if self._pending_conflicts:
            return self._pending_conflicts.pop(0)
        return None

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"MemoryBankAdapter(pending_conflicts={len(self._pending_conflicts)})"
