"""
MemoryBankAdapter

Adapter layer bridging the MemoryBank core with the host agent framework.
Responsible for prompt enhancement, context interception, and conflict routing.
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
from src.storage.query_graph import QueryGraphNode

logger = logging.getLogger(__name__)


class MemoryBankAdapter:
    """Bridge between MemoryBank and the external execution loop."""

    def __init__(self, memory_bank, retrieval_module) -> None:
        self.memory_bank = memory_bank
        self.retrieval = retrieval_module
        self._pending_conflicts: List[List[str]] = []
        self._merge_depth: int = 0  # Current recursive merge depth
        logger.info("MemoryBankAdapter initialized")

    # Public API

    def has_pending_conflicts(self) -> bool:
        """Check if there are pending conflicts to resolve."""
        result = len(self._pending_conflicts) > 0
        logger.debug(f"has_pending_conflicts() = {result}, queue size = {len(self._pending_conflicts)}")
        return result

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
        return prompt

    def intercept_context(self, context: str, task_type: str, insight_doc) -> None:
        """Convert tool output into structured memory or resolve conflicts."""
        if task_type == "CROSS_VALIDATE":
            self._handle_conflict_resolution(context, insight_doc)
        else:
            self._handle_normal_task(context, insight_doc)

    # Prompt helpers

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

        # Extract keywords: supports Chinese, English, numbers, filters stopwords
        import re
        words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z0-9]{2,}', query_text)
        stopwords = {'the', 'a', 'an', 'for', 'to', 'of', 'in', 'on', 'at', 'is', 'are'}
        query_keywords = [w for w in words if w.lower() not in stopwords][:10]
        if not query_keywords:
            query_keywords = [query_text]

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

    # Conflict resolution

    def _handle_conflict_resolution(self, validation_result: str, insight_doc) -> None:
        print("\nResolving conflicts...")

        logger.info(f"_handle_conflict_resolution called, queue size = {len(self._pending_conflicts)}")

        # Recursive depth check
        self._merge_depth += 1
        try:
            if self._merge_depth > self.memory_bank.config.MAX_MERGE_DEPTH:
                logger.warning(
                    f"Max merge depth ({self.memory_bank.config.MAX_MERGE_DEPTH}) reached. "
                    f"Stopping conflict resolution to prevent infinite recursion."
                )
                print(f"Maximum merge depth reached. Some conflicts may remain unresolved.")
                self._pending_conflicts.clear()  # Clear queue to avoid further processing
                return

            conflict_ids = self._get_conflicting_node_ids()
            if not conflict_ids:
                logger.warning("No pending conflicts found; skipping resolution.")
                logger.warning(f"Current queue state: {self._pending_conflicts}")
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
                        merge_description=node.merge_description,
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
                merge_description=integration_output.merge_description,
                links=[],
            )

            self.memory_bank.graph_ops.merge_nodes(conflict_ids, new_node)
            self.retrieval.mark_index_dirty()

            # Create interaction tree entry for cross_validate operation (save full validation context)
            self.memory_bank.interaction_tree.add_entry(new_node.id, validation_result)

            # Re-analyze relationships for merged node
            logger.info("Re-analyzing relationships for merged node...")

            # Use hybrid_retrieval to retrieve candidate nodes
            candidates = self.retrieval.hybrid_retrieval(
                query_embedding=new_node.embedding,
                query_keywords=new_node.keywords,
                graph=self.memory_bank.query_graph,
                exclude_ids={new_node.id}
            )

            if candidates:
                # Prepare Analysis Agent input
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

                # Call Analysis Agent to analyze relationships
                analysis_input = AnalysisInput(
                    new_node=new_node_info,
                    candidate_nodes=candidate_nodes
                )
                analysis_output = self.memory_bank.analysis_agent.run(analysis_input)

                # Process analysis results
                for rel in analysis_output.relationships:
                    if rel.relationship == "conflict":
                        # Found new conflict, add to queue (don't process immediately, let Planning Agent handle next round)
                        new_conflict = [new_node.id, rel.existing_node_id]
                        self._pending_conflicts.append(new_conflict)
                        logger.warning(
                            f"New conflict detected after merge: {new_node.id[:8]}... <-> {rel.existing_node_id[:8]}... "
                            f"Reason: {rel.conflict_description}"
                        )
                    elif rel.relationship == "related":
                        # Establish related edge
                        self.memory_bank.graph_ops.add_edge(new_node.id, rel.existing_node_id)
                        logger.info(
                            f"Established relationship: {new_node.id[:8]}... <-> {rel.existing_node_id[:8]}..."
                        )
                    # unrelated cases don't need handling, keep isolated

            if insight_doc.current_task:
                pending_desc = insight_doc.current_task
                # Use full merged summary as validation context
                validation_context = merged["summary"]

                insight_doc.add_completed_task(
                    task_type=TaskType.CROSS_VALIDATE,
                    description=pending_desc,
                    status="Success",
                    context=validation_context,
                )

            planning_output = self.memory_bank.planning_agent.run(
                PlanningInput(
                    insight_doc=insight_doc,
                    new_memory_nodes=[{  # Pass new merged node info to Planning Agent
                        "id": new_node.id,
                        "context": new_node.context,
                        "keywords": new_node.keywords,
                        "summary": new_node.summary,  # Include full validation result
                        "merge_description": new_node.merge_description,  # Include merge description
                    }],
                    conflict_notification=None,
                )
            )
            insight_doc.task_goal = planning_output.task_goal
            insight_doc.completed_tasks = planning_output.completed_tasks
            insight_doc.current_task = planning_output.current_task

            print(f"  Merged {len(conflict_ids)} conflicting node(s)")
            print("Conflict resolved\n")

        finally:
            # Recursive depth management
            self._merge_depth -= 1
            # If conflict queue is empty, reset merge depth
            if len(self._pending_conflicts) == 0:
                self._merge_depth = 0
                logger.info("All conflicts resolved, merge depth reset to 0")

    # Normal task handling

    def _handle_normal_task(self, context: str, insight_doc) -> None:
        print("\nOrganizing memory...")

        # Extract current task (if no task, use task_goal)
        current_task = insight_doc.current_task if insight_doc.current_task else insight_doc.task_goal

        print("  Classification Agent processing...")
        classification_output = self.memory_bank.classification_agent.run(
            ClassificationInput(
                context=context,
                task_goal=insight_doc.task_goal,
                current_task=current_task
            )
        )

        num_clusters = len(classification_output.clusters)
        print(f"  Found {num_clusters} topic cluster(s)\n")

        new_nodes: List[QueryGraphNode] = []
        conflicts: List[Dict[str, str]] = []
        total_relationships = 0

        for idx, cluster in enumerate(classification_output.clusters, 1):
            print(f"  Structure Agent processing... (cluster {idx}/{num_clusters})")

            structure_output = self.memory_bank.structure_agent.run(
                StructureInput(
                    content=cluster.content,
                    context=cluster.context,
                    keywords=cluster.keywords,
                    current_task=current_task,
                )
            )

            node = self.memory_bank._create_node(
                summary=structure_output.summary,
                context=cluster.context,
                keywords=cluster.keywords,
            )
            self.memory_bank.graph_ops.add_node(node)
            new_nodes.append(node)
            self.retrieval.mark_index_dirty()

            candidates = self.retrieval.hybrid_retrieval(
                query_embedding=node.embedding,
                query_keywords=node.keywords,
                graph=self.memory_bank.query_graph,
                exclude_ids={node.id},
            )

            if candidates:
                print(f"  Analysis Agent processing... (node {idx}/{num_clusters})")
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
                    continue  # skip related updates when conflict exists

                for rel in analysis_output.relationships:
                    if rel.relationship != "related":
                        continue

                    total_relationships += 1
                    if (
                        self.memory_bank.query_graph.has_node(node.id)
                        and self.memory_bank.query_graph.has_node(rel.existing_node_id)
                    ):
                        self.memory_bank.graph_ops.add_edge(node.id, rel.existing_node_id)

            # Save full context (full content classified by Classification Agent)
            self.memory_bank.interaction_tree.add_entry(node.id, cluster.content)

        print(f"  Created {len(new_nodes)} memory node(s)")
        print(f"  Established {total_relationships} relationship(s), {len(conflicts)} conflict(s)\n")

        conflict_notification = None
        if conflicts:
            # Use union-find to detect chain conflicts and group them
            conflict_groups = self._find_conflict_groups(conflicts)

            if conflict_groups:
                # Create conflict notification for Planning Agent (use first conflict group)
                first_group = conflict_groups[0]
                # Find conflict description for first group (use any conflict in group)
                first_description = "Conflict"
                for conflict in conflicts:
                    if conflict["new"] in first_group and conflict["existing"] in first_group:
                        first_description = conflict["description"]
                        break

                conflict_notification = ConflictNotification(
                    conflicting_node_ids=first_group,
                    conflict_description=first_description,
                )

                # Add all conflict groups to pending queue
                for group in conflict_groups:
                    if group not in self._pending_conflicts:
                        self._pending_conflicts.append(group)

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

        insight_doc.task_goal = planning_output.task_goal
        insight_doc.completed_tasks = planning_output.completed_tasks
        insight_doc.current_task = planning_output.current_task

        print("Memory organized\n")

    # Helpers

    def _find_conflict_groups(self, conflicts: List[Dict[str, str]]) -> List[List[str]]:
        """
        Use Union-Find algorithm to detect chain conflicts and group transitive conflicts

        Example: A-B conflict, B-C conflict -> A, B, C should be grouped together

        Args:
            conflicts: Conflict list, each conflict contains "new" and "existing" node IDs

        Returns:
            List of conflict groups, each group contains all transitive conflict node IDs
        """
        if not conflicts:
            return []

        # Union-Find: parent[node] = parent node of node
        parent: Dict[str, str] = {}

        def find(node: str) -> str:
            """Find root node (with path compression)"""
            if node not in parent:
                parent[node] = node
            if parent[node] != node:
                parent[node] = find(parent[node])  # Path compression
            return parent[node]

        def union(node1: str, node2: str) -> None:
            """Merge two nodes' sets"""
            root1 = find(node1)
            root2 = find(node2)
            if root1 != root2:
                parent[root2] = root1  # Set root2's parent to root1

        # 1. Build Union-Find: process all conflict pairs
        for conflict in conflicts:
            union(conflict["new"], conflict["existing"])

        # 2. Group by root node
        groups: Dict[str, List[str]] = {}
        for node in parent.keys():
            root = find(node)
            if root not in groups:
                groups[root] = []
            groups[root].append(node)

        # 3. Return all conflict groups (filter single-node groups)
        result = [group for group in groups.values() if len(group) > 1]
        logger.info(f"Detected {len(result)} conflict group(s) from {len(conflicts)} conflict(s)")

        return result

    def _get_conflicting_node_ids(self) -> Optional[List[str]]:
        if self._pending_conflicts:
            result = self._pending_conflicts.pop(0)
            logger.info(f"Popped conflict group: {[nid[:8] for nid in result]}, remaining queue size = {len(self._pending_conflicts)}")
            return result
        logger.warning("_get_conflicting_node_ids: Queue is empty")
        return None

    def __repr__(self) -> str:
        return f"MemoryBankAdapter(pending_conflicts={len(self._pending_conflicts)})"
