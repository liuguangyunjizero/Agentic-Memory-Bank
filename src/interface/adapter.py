"""
Interface layer connecting memory bank operations to external agent frameworks.
Manages prompt construction, context processing, and conflict workflow orchestration.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from src.agents.analysis_agent import AnalysisInput, NodeInfo
from src.agents.integration_agent import IntegrationInput, NodeWithNeighbors
from src.agents.planning_agent import ConflictNotification, PlanningInput
from src.agents.structure_agent import StructureInput
from src.storage.insight_doc import TaskType
from src.storage.query_graph import QueryGraphNode

logger = logging.getLogger(__name__)


class MemoryBankAdapter:
    """Coordinates memory operations and ReAct agent communication."""

    def __init__(self, memory_bank, retrieval_module) -> None:
        self.memory_bank = memory_bank
        self.retrieval = retrieval_module
        self._pending_conflicts: List[Dict[str, Any]] = []
        logger.info("MemoryBankAdapter initialized")

    def has_pending_conflicts(self) -> bool:
        """Check whether unresolved conflicts remain in the queue."""
        result = len(self._pending_conflicts) > 0
        logger.debug(f"has_pending_conflicts() = {result}, queue size = {len(self._pending_conflicts)}")
        return result

    def enhance_prompt(self, insight_doc) -> str:
        """Build structured prompt containing task state and relevant memories for ReAct agent."""
        if insight_doc is None:
            return "<task>\nNo active task\n</task>\n\n<memory>\nNo related memory\n</memory>"

        task_section = self._build_task_section(insight_doc)
        memory_section = self._build_memory_section(insight_doc)

        prompt = (
            f"<task>\n{task_section}\n</task>\n\n"
            f"<memory>\n{memory_section}\n</memory>\n\n"
            "Please continue with the next step."
        )
        return prompt

    def intercept_context(self, context: str, task_type: str, insight_doc) -> None:
        """Route completed task output to either conflict resolution or normal memory storage."""
        if task_type == "CROSS_VALIDATE":
            self._handle_conflict_resolution(context, insight_doc)
        else:
            self._handle_normal_task(context, insight_doc)

    def _build_task_section(self, insight_doc) -> str:
        """Format current goal, history, and active task into readable section."""
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
        """Retrieve and format memories relevant to the current task using hybrid search."""
        if not insight_doc.current_task:
            return "No related memory"

        query_text = insight_doc.current_task
        query_embedding = self.memory_bank.embedding_module.compute_embedding(query_text)

        query_keywords = insight_doc.current_task_keywords or [query_text]

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

    def _handle_conflict_resolution(self, validation_result: str, insight_doc) -> None:
        """
        Execute full conflict resolution pipeline: merge nodes, re-analyze, and update plan.
        Uses validation evidence to guide integration decisions.
        """
        print("\nResolving conflicts...")

        logger.info(f"_handle_conflict_resolution called, queue size = {len(self._pending_conflicts)}")

        conflict_payload = self._get_conflicting_node_ids()
        if not conflict_payload:
            logger.warning("No pending conflicts found; skipping resolution.")
            logger.warning(f"Current queue state: {self._pending_conflicts}")
            return
        conflict_ids = conflict_payload.get("node_ids", [])
        conflict_reason = conflict_payload.get("description", "Conflict detected")

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
                    merge_description=node.merge_description,
                    core_information=node.core_information,
                    supporting_evidence=node.supporting_evidence,
                    structure_summary=node.structure_summary,
                    acquisition_logic=node.acquisition_logic,
                    neighbors=[
                        {"id": n.id, "context": n.context, "keywords": n.keywords}
                        for n in neighbours
                    ],
                )
            )

        if not nodes_to_merge:
            logger.error("Unable to load conflict nodes; aborting integration.")
            return

        affected_tasks = insight_doc.get_tasks_for_nodes(conflict_ids)
        for task_desc in affected_tasks:
            insight_doc.mark_task_failed(task_desc, conflict_reason)

        integration_output = self.memory_bank.integration_agent.run(
            IntegrationInput(nodes_to_merge=nodes_to_merge, validation_result=validation_result)
        )

        merged = integration_output.merged_node
        embedding_text = (
            f"{merged.get('summary', '')} "
            f"{merged.get('context', '')} "
            f"{' '.join(merged.get('keywords', []))} "
            f"{merged.get('core_information', '')} "
            f"{merged.get('structure_summary', '')}"
        )
        new_node = QueryGraphNode(
            id=str(uuid.uuid4()),
            summary=merged["summary"],
            context=merged["context"],
            keywords=merged["keywords"],
            embedding=self.memory_bank.embedding_module.compute_embedding(embedding_text),
            timestamp=time.time(),
            merge_description=integration_output.merge_description,
            core_information=merged.get("core_information", merged["summary"]),
            supporting_evidence=merged.get("supporting_evidence", ""),
            structure_summary=merged.get("structure_summary", merged["summary"]),
            acquisition_logic=merged.get("acquisition_logic"),
            links=[],
        )

        self.memory_bank.graph_ops.merge_nodes(conflict_ids, new_node)
        self.retrieval.mark_index_dirty()

        self.memory_bank.interaction_tree.add_entry(new_node.id, validation_result)

        logger.info("Re-analyzing relationships for merged node...")

        candidates = self.retrieval.hybrid_retrieval(
            query_embedding=new_node.embedding,
            query_keywords=new_node.keywords,
            graph=self.memory_bank.query_graph,
            exclude_ids={new_node.id}
        )

        if candidates:
            new_node_info = NodeInfo(
                id=new_node.id,
                summary=new_node.summary,
                context=new_node.context,
                keywords=new_node.keywords,
            )

            candidate_nodes = [
                NodeInfo(
                    id=node.id,
                    summary=node.summary,
                    context=node.context,
                    keywords=node.keywords,
                )
                for node in candidates
            ]

            analysis_input = AnalysisInput(
                new_node=new_node_info,
                candidate_nodes=candidate_nodes
            )
            analysis_output = self.memory_bank.analysis_agent.run(analysis_input)

            for rel in analysis_output.relationships:
                if rel.relationship == "conflict":
                    payload = {
                        "node_ids": [new_node.id, rel.existing_node_id],
                        "description": rel.conflict_description or "Conflict detected after merge"
                    }
                    self._pending_conflicts.append(payload)
                    logger.warning(
                        f"New conflict detected after merge: {new_node.id[:8]}... <-> {rel.existing_node_id[:8]}... "
                        f"Reason: {rel.conflict_description}"
                    )
                elif rel.relationship == "related":
                    self.memory_bank.graph_ops.add_edge(new_node.id, rel.existing_node_id)
                    logger.info(
                        f"Established relationship: {new_node.id[:8]}... <-> {rel.existing_node_id[:8]}..."
                    )

        if insight_doc.current_task:
            pending_desc = insight_doc.current_task
            validation_context = merged.get("core_information") or merged["summary"]

            insight_doc.add_completed_task(
                task_type=TaskType.CROSS_VALIDATE,
                description=pending_desc,
                status="Success",
                context=validation_context,
            )

        planning_output = self.memory_bank.planning_agent.run(
            PlanningInput(
                insight_doc=insight_doc,
                new_memory_nodes=[{
                    "id": new_node.id,
                    "context": new_node.context,
                    "keywords": new_node.keywords,
                    "summary": new_node.summary,
                    "merge_description": new_node.merge_description,
                    "core_information": new_node.core_information,
                    "supporting_evidence": new_node.supporting_evidence,
                    "structure_summary": new_node.structure_summary,
                }],
                conflict_notification=None,
            )
        )
        updated_tasks = insight_doc.enforce_failed_statuses(planning_output.completed_tasks)
        insight_doc.task_goal = planning_output.task_goal
        insight_doc.completed_tasks = updated_tasks
        insight_doc.current_task = planning_output.current_task
        insight_doc.current_task_keywords = planning_output.current_task_keywords

        print(f"  Merged {len(conflict_ids)} conflicting node(s)")
        print("Conflict resolved\n")

    def _handle_normal_task(self, context: str, insight_doc) -> None:
        """
        Process tool output into structured memory: extract, analyze, detect conflicts, and plan next step.
        Creates new node and establishes relationships with existing memories.
        """
        print("\nOrganizing memory...")

        current_task = insight_doc.current_task if insight_doc.current_task else insight_doc.task_goal

        print("  Structure Agent processing...")
        structure_output = self.memory_bank.structure_agent.run(
            StructureInput(content=context)
        )

        node = self.memory_bank._create_node(structure_output)
        self.memory_bank.graph_ops.add_node(node)
        self.retrieval.mark_index_dirty()
        self.memory_bank.interaction_tree.add_entry(node.id, context)
        insight_doc.register_task_node(current_task, node.id)

        conflicts: List[Dict[str, str]] = []
        total_relationships = 0

        candidates = self.retrieval.hybrid_retrieval(
            query_embedding=node.embedding,
            query_keywords=node.keywords,
            graph=self.memory_bank.query_graph,
            exclude_ids={node.id},
        )

        if candidates:
            print("  Analysis Agent processing...")
            analysis_output = self.memory_bank.analysis_agent.run(
                AnalysisInput(
                    new_node=NodeInfo(
                        id=node.id,
                        summary=node.summary,
                        context=node.context,
                        keywords=node.keywords,
                        merge_description=node.merge_description,
                    ),
                    candidate_nodes=[
                        NodeInfo(
                            id=c.id,
                            summary=c.summary,
                            context=c.context,
                            keywords=c.keywords,
                            merge_description=c.merge_description,
                        )
                        for c in candidates
                    ],
                )
            )

            conflict_rels = [
                rel for rel in analysis_output.relationships if rel.relationship == "conflict"
            ]
            if conflict_rels:
                for rel in conflict_rels:
                    conflicts.append(
                        {
                            "new": node.id,
                            "existing": rel.existing_node_id,
                            "description": rel.conflict_description or "Conflict",
                        }
                    )
            else:
                for rel in analysis_output.relationships:
                    if rel.relationship != "related":
                        continue

                    total_relationships += 1
                    if (
                        self.memory_bank.query_graph.has_node(node.id)
                        and self.memory_bank.query_graph.has_node(rel.existing_node_id)
                    ):
                        self.memory_bank.graph_ops.add_edge(node.id, rel.existing_node_id)

        print("  Created 1 memory node(s)")
        print(f"  Established {total_relationships} relationship(s), {len(conflicts)} conflict(s)\n")

        conflict_notification = None
        if conflicts:
            conflict_groups = self._find_conflict_groups(conflicts)

            if conflict_groups:
                first_group = conflict_groups[0]
                first_description = "Conflict"
                for conflict in conflicts:
                    if conflict["new"] in first_group and conflict["existing"] in first_group:
                        first_description = conflict["description"]
                        break

                conflict_notification = ConflictNotification(
                    conflicting_node_ids=first_group,
                    conflict_description=first_description,
                )

                for group in conflict_groups:
                    payload = {
                        "node_ids": group,
                        "description": first_description
                        if group == first_group
                        else self._describe_conflict_group(conflicts, group)
                    }
                    if payload not in self._pending_conflicts:
                        self._pending_conflicts.append(payload)

                for group in conflict_groups:
                    reason = self._describe_conflict_group(conflicts, group)
                    affected_tasks = insight_doc.get_tasks_for_nodes(group)
                    for task_desc in affected_tasks:
                        insight_doc.mark_task_failed(task_desc, reason)

        planning_output = self.memory_bank.planning_agent.run(
            PlanningInput(
                insight_doc=insight_doc,
                new_memory_nodes=[
                    {
                        "id": node.id,
                        "context": node.context,
                        "keywords": node.keywords,
                        "summary": node.summary,
                        "core_information": node.core_information,
                        "supporting_evidence": node.supporting_evidence,
                        "structure_summary": node.structure_summary,
                    }
                ],
                conflict_notification=conflict_notification,
            )
        )

        updated_tasks = insight_doc.enforce_failed_statuses(planning_output.completed_tasks)
        insight_doc.task_goal = planning_output.task_goal
        insight_doc.completed_tasks = updated_tasks
        insight_doc.current_task = planning_output.current_task
        insight_doc.current_task_keywords = planning_output.current_task_keywords

        print("Memory organized\n")

    def _find_conflict_groups(self, conflicts: List[Dict[str, str]]) -> List[List[str]]:
        """
        Detect transitive conflict chains using Union-Find algorithm.
        Groups all mutually conflicting nodes together for simultaneous resolution.
        Example: if A conflicts with B and B conflicts with C, all three are grouped.
        """
        if not conflicts:
            return []

        parent: Dict[str, str] = {}

        def find(node: str) -> str:
            """Find root with path compression for efficiency."""
            if node not in parent:
                parent[node] = node
            if parent[node] != node:
                parent[node] = find(parent[node])
            return parent[node]

        def union(node1: str, node2: str) -> None:
            """Merge two nodes into same conflict group."""
            root1 = find(node1)
            root2 = find(node2)
            if root1 != root2:
                parent[root2] = root1

        for conflict in conflicts:
            union(conflict["new"], conflict["existing"])

        groups: Dict[str, List[str]] = {}
        for node in parent.keys():
            root = find(node)
            if root not in groups:
                groups[root] = []
            groups[root].append(node)

        result = [group for group in groups.values() if len(group) > 1]
        logger.info(f"Detected {len(result)} conflict group(s) from {len(conflicts)} conflict(s)")

        return result

    @staticmethod
    def _describe_conflict_group(conflicts: List[Dict[str, str]], group: List[str]) -> str:
        """Extract first conflict description matching nodes in the group."""
        for conflict in conflicts:
            if conflict["new"] in group and conflict["existing"] in group:
                return conflict["description"]
        return "Conflict detected among nodes"

    def _get_conflicting_node_ids(self) -> Optional[Dict[str, Any]]:
        """Pop next conflict group from queue for resolution."""
        if self._pending_conflicts:
            payload = self._pending_conflicts.pop(0)
            node_ids = payload.get("node_ids", [])
            logger.info(
                f"Popped conflict group: {[nid[:8] for nid in node_ids]}, remaining queue size = {len(self._pending_conflicts)}"
            )
            return payload
        logger.warning("_get_conflicting_node_ids: Queue is empty")
        return None


    def __repr__(self) -> str:
        return f"MemoryBankAdapter(pending_conflicts={len(self._pending_conflicts)})"
