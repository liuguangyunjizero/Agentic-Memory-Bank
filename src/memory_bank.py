"""
Agentic Memory Bank Core Class

Integrates all components to provide complete memory management functionality.
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
from src.config import Config
from src.prompts.agent_prompts import REACT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class MemoryBank:
    """Agentic Memory Bank main class"""

    def __init__(self, config: Config = None):
        """
        Initialize Memory Bank

        Args:
            config: Configuration object (optional, uses default if not provided)
        """
        logger.info("Initializing Agentic Memory Bank...")

        # Initialize configuration
        self.config = config or Config()

        # Initialize storage layer
        self.query_graph = QueryGraph()
        self.interaction_tree = InteractionTree()
        self.insight_doc = None  # Created separately for each task

        # Initialize tools
        self.llm_client = LLMClient.from_config(self.config)

        # Initialize hardcoded modules
        self.embedding_module = EmbeddingModule.from_config(self.config)
        self.retrieval_module = RetrievalModule(
            alpha=self.config.RETRIEVAL_ALPHA,
            k=self.config.RETRIEVAL_K
        )
        self.graph_ops = GraphOperations(self.query_graph, self.interaction_tree)

        # Initialize Agents
        self.classification_agent = ClassificationAgent.from_config(
            self.llm_client, self.config
        )
        self.structure_agent = StructureAgent.from_config(self.llm_client, self.config)
        self.analysis_agent = AnalysisAgent.from_config(self.llm_client, self.config)
        self.integration_agent = IntegrationAgent.from_config(self.llm_client, self.config)
        self.planning_agent = PlanningAgent.from_config(self.llm_client, self.config)

        # Initialize Interface layer
        self.deep_retrieval_tool = DeepRetrievalTool(self.interaction_tree)

        # Search tool: Use real search if Serper API key is configured
        search_api_key = self.config.SERPER_API_KEY
        if not search_api_key or search_api_key == "your-serper-api-key-here":
            raise ValueError(
                "Serper API key not configured. Please set SERPER_API_KEY in .env file.\n"
                "Sign up at: https://serper.dev/"
            )

        self.search_tool = SearchTool(search_api_key=search_api_key)

        # Visit tool: Use Jina Reader API (required)
        jina_api_key = self.config.JINA_API_KEY
        if not jina_api_key or jina_api_key == "your-jina-api-key-here":
            raise ValueError(
                "Jina API key not configured. Please set JINA_API_KEY in .env file.\n"
                "Sign up at: https://jina.ai/"
            )

        self.visit_tool = VisitTool(
            llm_client=self.llm_client,
            jina_api_key=jina_api_key,
            temperature=self.config.VISIT_EXTRACTION_TEMPERATURE,
            top_p=self.config.VISIT_EXTRACTION_TOP_P
        )

        # Initialize Adapter
        self.adapter = MemoryBankAdapter(self, self.retrieval_module)

        # Initialize ReAct Agent
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
        Execute a single task

        Args:
            user_input: User input (may contain context + question)

        Returns:
            Task result: {
                "answer": str,
                "insight_doc": dict,
                "stats": dict
            }
        """
        logger.info("=" * 60)
        logger.info(f"Starting new task: {user_input}")
        logger.info("=" * 60)

        try:
            # Use simplified input parsing
            text_context, question, is_context_only = self._parse_user_input_simple(user_input)

            if is_context_only:
                # Context loading only, skip full task workflow
                print("\n" + "=" * 80)
                print("  Loading Context Only Mode")
                print("=" * 80)
                result = self._load_context_only(text_context)
                return result

            # 1. Initialization phase (normal task workflow)
            print("\n" + "=" * 80)
            print("  Memory Bank Initialized")
            print("=" * 80)
            enhanced_prompt = self._initialize(user_input)
            iterations = 0

            # Display task goal and initial state
            if self.insight_doc:
                print(f"Task: {self.insight_doc.task_goal}")
                print(f"Current Subtask: {self.insight_doc.current_task if self.insight_doc.current_task else '(none)'}")
                print("=" * 80)

            # 2. Execution loop
            answer = None
            last_react_result = None  # Save last ReAct result
            max_iterations = self.config.MAX_LLM_CALL_PER_RUN
            while not self._should_terminate() and iterations < max_iterations:
                iterations += 1
                print(f"\n{'=' * 80}")
                print(f"  Iteration {iterations}")
                print("=" * 80)

                # Display current task status (simplified)
                if self.insight_doc and self.insight_doc.current_task:
                    print(f"Current: {self.insight_doc.current_task}")
                    print(f"Completed: {len(self.insight_doc.completed_tasks)} | Memory nodes: {self.query_graph.get_node_count()}")

                # 2.1 ReAct execution
                react_result = self.react_agent.run(enhanced_prompt)

                # 2.2 Context interception: Convert all valuable content to memory
                if self.insight_doc and self.insight_doc.current_task:
                    messages = react_result.get("messages", [])

                    # Extract complete context (including thoughts, tool calls, answers and system messages)
                    full_context = "\n\n".join([
                        f"[{m.get('role', 'unknown')}]:\n{m.get('content', '')}"
                        for m in messages
                        if m.get('content', '').strip()
                    ])

                    # Check if there's valuable content
                    has_valuable_content = bool(full_context and full_context.strip())

                    if has_valuable_content:
                        try:
                            # Determine task type
                            has_conflict = self.adapter.has_pending_conflicts()
                            task_type = "CROSS_VALIDATE" if has_conflict else "NORMAL"

                            # Save all valuable content (direct answers, tool calls, final answers all saved)
                            self.adapter.intercept_context(full_context, task_type, self.insight_doc)

                        except Exception as e:
                            print(f"Error: {str(e)}")
                            logger.error(f"Context interception failed: {str(e)}")
                            import traceback
                            traceback.print_exc()

                # 2.3 Check if there are pending tasks (if none, task is complete)
                if not self.insight_doc or not self.insight_doc.current_task:
                    print("\n[DONE] All tasks complete")
                    break

                # 2.4 Enhance next round Prompt (based on new task state)
                if self.insight_doc.current_task:
                    print(f"\nNext: {self.insight_doc.current_task}")
                enhanced_prompt = self.adapter.enhance_prompt(self.insight_doc)

            # 3. Statistics
            final_insight_doc = self.insight_doc.to_dict() if self.insight_doc else None

            stats = {
                "iterations": iterations,
                "graph_nodes": self.query_graph.get_node_count(),
                "graph_edges": self.query_graph.get_edge_count(),
                "tree_entries": self.interaction_tree.get_total_entries(),
                "completed_tasks": len(self.insight_doc.completed_tasks) if self.insight_doc else 0,
                "current_task": self.insight_doc.current_task if self.insight_doc else ""
            }

            logger.debug("\n" + "=" * 60)
            logger.debug("Task completed")
            logger.debug(f"Stats: {stats}")
            logger.debug("=" * 60)

            # Return simplified result (mainly for debugging and testing)
            return {
                "insight_doc": final_insight_doc,
                "stats": stats
            }

        except Exception as e:
            logger.error(f"Task execution failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _load_context_only(self, text_context: str) -> Dict[str, Any]:
        """
        Load context only, without executing full task workflow

        Args:
            text_context: Context text to load

        Returns:
            Loading result: {
                "answer": str,
                "stats": dict
            }
        """
        logger.info("Loading context only mode")

        # Parse user input (actually already have text_context, but for consistency)
        doc_id = str(uuid.uuid4())

        # Check if there's text context
        if not text_context:
            return {"answer": "No context provided", "stats": {}}

        # 1. Classification/Clustering (same as normal workflow)
        print("\n[Classification] Calling Classification Agent...")
        classification_output = self.classification_agent.run(ClassificationInput(
            context=text_context,
            task_goal="Context loading"  # Virtual task goal
        ))

        # 2-3. Process each cluster (same as normal workflow, but no planning)
        new_nodes = []
        all_conflicts = []  # Collect all conflicts

        for i, cluster in enumerate(classification_output.clusters):

            # 4. Structuring
            print(f"\n  [{i+1}/{len(classification_output.clusters)}] Calling Structure Agent...")
            structure_output = self.structure_agent.run(StructureInput(
                content=cluster.content,
                context=cluster.context,
                keywords=cluster.keywords,
                current_task="Context loading"  # Virtual current task
            ))

            # 5. Assemble node and add to graph
            node = self._create_node(
                summary=structure_output.summary,
                context=cluster.context,
                keywords=cluster.keywords
            )
            self.graph_ops.add_node(node)
            new_nodes.append(node)

            # 6. Mark index as dirty (rebuild on next retrieval)
            self.retrieval_module.mark_index_dirty()

            # 7. Retrieve similar nodes (for analyzing potential conflicts)
            candidates = self.retrieval_module.hybrid_retrieval(
                query_embedding=node.embedding,
                query_keywords=node.keywords,
                graph=self.query_graph,
                exclude_ids={node.id}
            )

            # 8. Analyze relationships (if candidate nodes found)
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

                # Process relationships (record conflict info, add related edges)
                for rel in analysis_output.relationships:
                    if rel.relationship == "conflict":
                        # Record conflict info (if config enables conflict reporting)
                        if self.config.REPORT_CONFLICTS_IN_CONTEXT_LOADING:
                            all_conflicts.append({
                                "new_node_id": node.id,
                                "existing_node_id": rel.existing_node_id,
                                "description": rel.conflict_description,
                                "new_context": node.context,
                                "existing_context": self.query_graph.get_node(rel.existing_node_id).context
                            })
                    elif rel.relationship == "related":
                        self.graph_ops.add_edge(node.id, rel.existing_node_id)

            # 9. Create Interaction Tree Entry (save complete content)
            self.interaction_tree.add_entry(node.id, cluster.content)

        # Statistics
        stats = {
            "nodes_added": len(new_nodes),
            "total_nodes": self.query_graph.get_node_count(),
            "total_edges": self.query_graph.get_edge_count(),
            "conflicts_detected": len(all_conflicts)
        }

        logger.info(f"Context loaded successfully: {len(new_nodes)} nodes added, {len(all_conflicts)} conflicts detected")

        # Generate return message
        base_message = f"Context loaded successfully. Added {len(new_nodes)} nodes to memory."

        # If conflicts detected, add conflict report
        if all_conflicts and self.config.REPORT_CONFLICTS_IN_CONTEXT_LOADING:
            conflict_report = f"\n\nDetected {len(all_conflicts)} potential conflict(s):\n"
            for i, conflict in enumerate(all_conflicts[:3], 1):  # Show first 3 conflicts at most
                conflict_report += f"\n{i}. {conflict['description']}"
                conflict_report += f"\n   New: {conflict['new_context']}"
                conflict_report += f"\n   Existing: {conflict['existing_context']}\n"

            if len(all_conflicts) > 3:
                conflict_report += f"\n... and {len(all_conflicts) - 3} more conflict(s)."

            base_message += conflict_report

        return {
            "answer": base_message,
            "stats": stats,
            "conflicts": all_conflicts  # For main.py use
        }

    def _initialize(self, user_input: str) -> str:
        """
        Initialization phase

        Workflow:
        1. Parse user input
        2. Multimodal temporary storage
        3. Classification -> Structuring -> Retrieval -> Analysis -> Build edges
        4. Plan next steps
        5. Enhance Prompt

        Args:
            user_input: User input

        Returns:
            Enhanced Prompt
        """
        # 1. Parse user input
        text_context, question, _ = self._parse_user_input_simple(user_input)
        doc_id = str(uuid.uuid4())

        # 2. If there's text context, load it into memory first before processing question
        if text_context:
            logger.info("Loading text context into memory before processing question...")
            # Directly call _load_context_only logic to process context
            # But don't return result, continue processing question
            self._load_context_only(text_context)

        # 3. Initialize Insight Doc and plan
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

    def _create_node(self, summary: str, context: str, keywords: List[str]) -> QueryGraphNode:
        """
        Create Query Graph node

        Args:
            summary: Summary
            context: Context
            keywords: List of keywords

        Returns:
            QueryGraphNode instance
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
            links=[]  # Should be list, not set
        )

    def _parse_user_input_simple(self, user_input: str) -> Tuple[str, str, bool]:
        """
        Simplified input parsing - provides clearer user experience

        Supports three modes:
        1. Pure question - directly input question
        2. Starting with "Context:" - context loading only
        3. Contains "Question:" - context + question mode

        Args:
            user_input: User input

        Returns:
            (text_context, question, is_context_only)
        """
        user_input = user_input.strip()

        # Mode 1: Context only
        if user_input.lower().startswith("context:") or user_input.startswith("上下文："):
            # Remove prefix, get context content
            if user_input.lower().startswith("context:"):
                text_context = user_input[8:].strip()
            else:
                text_context = user_input[4:].strip()
            return text_context, "", True

        # Mode 2: Context + question (check if contains Question marker)
        if "question:" in user_input.lower() or "问题：" in user_input:
            # Parse context and question
            text_context = ""
            question = ""

            # Try to separate context and question
            if "\nquestion:" in user_input.lower():
                parts = user_input.split("\nquestion:", 1)
                text_context = parts[0].strip()
                question = parts[1].strip()
            elif "\n问题：" in user_input:
                parts = user_input.split("\n问题：", 1)
                text_context = parts[0].strip()
                question = parts[1].strip()

            # Clean context prefix
            if text_context.lower().startswith("context:"):
                text_context = text_context[8:].strip()
            elif text_context.startswith("上下文："):
                text_context = text_context[4:].strip()

            return text_context, question, False

        # Mode 3: Pure question (no markers)
        return "", user_input, False


    def _should_terminate(self) -> bool:
        """
        Determine if should terminate

        Returns:
            Whether task should be terminated
        """
        if not self.insight_doc:
            return False

        # Terminate if no pending tasks
        return not self.insight_doc.current_task

    def export_memory(self, filepath: str):
        """
        Export memory to JSON file

        Args:
            filepath: Output file path
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
        Load memory from JSON file

        Args:
            filepath: Input file path
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
        """Return Memory Bank summary"""
        return (
            f"MemoryBank("
            f"nodes={self.query_graph.get_node_count()}, "
            f"edges={self.query_graph.get_edge_count()}, "
            f"entries={self.interaction_tree.get_total_entries()})"
        )

    def clear_memory(self) -> None:
        """
        Clear all memory (Query Graph, Interaction Tree, Insight Doc)

        Call this explicitly when:
        - Ending a session (user quits)
        - Starting a new unrelated conversation
        - Running out of memory

        Note: This does NOT affect exported memory files
        """
        logger.info("Clearing all memory...")

        self.query_graph.clear()
        self.retrieval_module.mark_index_dirty()
        self.interaction_tree.clear()
        self.insight_doc = None

        # Clear adapter's conflict queue
        if hasattr(self.adapter, '_pending_conflicts'):
            self.adapter._pending_conflicts.clear()
        # Reset merge depth
        if hasattr(self.adapter, '_merge_depth'):
            self.adapter._merge_depth = 0

        logger.info("Memory cleared successfully")
