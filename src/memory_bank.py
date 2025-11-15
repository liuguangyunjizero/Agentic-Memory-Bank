"""
Central orchestrator that coordinates all components of the memory system.
Manages the complete lifecycle from context ingestion through query execution.
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
from src.agents.structure_agent import StructureAgent, StructureInput, StructureOutput
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
    """
    Main controller that integrates storage layers, agents, and tools into a unified system.
    Handles initialization, execution loops, and memory persistence operations.
    """

    def __init__(self, config: Config = None):
        """
        Bootstrap the entire system by initializing all layers and components.
        Validates required API keys and creates necessary connections.
        """
        logger.info("Initializing Agentic Memory Bank...")

        self.config = config or Config()

        self.query_graph = QueryGraph()
        self.interaction_tree = InteractionTree()
        self.insight_doc = None

        self.llm_client = LLMClient.from_config(self.config)

        self.embedding_module = EmbeddingModule.from_config(self.config)
        self.retrieval_module = RetrievalModule(
            alpha=self.config.RETRIEVAL_ALPHA,
            k=self.config.RETRIEVAL_K
        )
        self.graph_ops = GraphOperations(self.query_graph, self.interaction_tree)

        self.classification_agent = ClassificationAgent.from_config(
            self.llm_client, self.config
        )
        self.structure_agent = StructureAgent.from_config(self.llm_client, self.config)
        self.analysis_agent = AnalysisAgent.from_config(self.llm_client, self.config)
        self.integration_agent = IntegrationAgent.from_config(self.llm_client, self.config)
        self.planning_agent = PlanningAgent.from_config(self.llm_client, self.config)

        self.deep_retrieval_tool = DeepRetrievalTool(self.interaction_tree)

        search_api_key = self.config.SERPER_API_KEY
        if not search_api_key or search_api_key == "your-serper-api-key-here":
            raise ValueError(
                "Serper API key not configured. Please set SERPER_API_KEY in .env file.\n"
                "Sign up at: https://serper.dev/"
            )

        self.search_tool = SearchTool(search_api_key=search_api_key)

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

        self.adapter = MemoryBankAdapter(self, self.retrieval_module)

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
        Main entry point that processes user input through the complete workflow.
        Handles both context-only loading and question-answering with iterative execution.
        Returns execution statistics and final state.
        """
        logger.info("=" * 60)
        logger.info(f"Starting new task: {user_input}")
        logger.info("=" * 60)

        try:
            text_context, question, is_context_only = self._parse_user_input_simple(user_input)

            if is_context_only:
                print("\n" + "=" * 80)
                print("  Loading Context Only Mode")
                print("=" * 80)
                result = self._load_context_only(text_context)
                return result

            print("\n" + "=" * 80)
            print("  Memory Bank Initialized")
            print("=" * 80)
            enhanced_prompt = self._initialize(user_input)
            iterations = 0

            if self.insight_doc:
                print(f"Task: {self.insight_doc.task_goal}")
                print(f"Current Subtask: {self.insight_doc.current_task if self.insight_doc.current_task else '(none)'}")
                print("=" * 80)

            answer = None
            last_react_result = None
            max_iterations = self.config.MAX_LLM_CALL_PER_RUN
            while not self._should_terminate() and iterations < max_iterations:
                iterations += 1
                print(f"\n{'=' * 80}")
                print(f"  Iteration {iterations}")
                print("=" * 80)

                if self.insight_doc and self.insight_doc.current_task:
                    print(f"Current: {self.insight_doc.current_task}")
                    print(f"Completed: {len(self.insight_doc.completed_tasks)} | Memory nodes: {self.query_graph.get_node_count()}")

                react_result = self.react_agent.run(enhanced_prompt)

                if self.insight_doc and self.insight_doc.current_task:
                    messages = react_result.get("messages", [])

                    full_context = "\n\n".join([
                        f"[{m.get('role', 'unknown')}]:\n{m.get('content', '')}"
                        for m in messages
                        if m.get('content', '').strip()
                    ])

                    has_valuable_content = bool(full_context and full_context.strip())

                    if has_valuable_content:
                        try:
                            has_conflict = self.adapter.has_pending_conflicts()
                            task_type = "CROSS_VALIDATE" if has_conflict else "NORMAL"

                            self.adapter.intercept_context(full_context, task_type, self.insight_doc)

                        except Exception as e:
                            print(f"Error: {str(e)}")
                            logger.error(f"Context interception failed: {str(e)}")
                            import traceback
                            traceback.print_exc()

                if not self.insight_doc or not self.insight_doc.current_task:
                    print("\n[DONE] All tasks complete")
                    break

                if self.insight_doc.current_task:
                    print(f"\nNext: {self.insight_doc.current_task}")
                enhanced_prompt = self.adapter.enhance_prompt(self.insight_doc)

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
        Process and store context without creating execution tasks.
        Segments the input, structures each segment, and builds the knowledge graph.
        """
        logger.info("Loading context only mode")

        doc_id = str(uuid.uuid4())

        if not text_context:
            return {"answer": "No context provided", "stats": {}}

        print("\n[Segmentation] Calling Classification Agent...")
        classification_output = self.classification_agent.run(ClassificationInput(
            context=text_context
        ))

        new_nodes = []

        for i, cluster in enumerate(classification_output.clusters):

            print(f"\n  [{i+1}/{len(classification_output.clusters)}] Calling Structure Agent...")
            structure_output = self.structure_agent.run(StructureInput(
                content=cluster.content
            ))

            node = self._create_node(structure_output)
            self.graph_ops.add_node(node)
            new_nodes.append(node)

            self.retrieval_module.mark_index_dirty()

            candidates = self.retrieval_module.hybrid_retrieval(
                query_embedding=node.embedding,
                query_keywords=node.keywords,
                graph=self.query_graph,
                exclude_ids={node.id}
            )

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

                for rel in analysis_output.relationships:
                    if rel.relationship == "related":
                        self.graph_ops.add_edge(node.id, rel.existing_node_id)

            self.interaction_tree.add_entry(node.id, cluster.content)

        stats = {
            "nodes_added": len(new_nodes),
            "total_nodes": self.query_graph.get_node_count(),
            "total_edges": self.query_graph.get_edge_count()
        }

        logger.info(f"Context loaded successfully: {len(new_nodes)} nodes added")

        return {
            "answer": f"Context loaded successfully. Added {len(new_nodes)} nodes to memory.",
            "stats": stats
        }

    def _initialize(self, user_input: str) -> str:
        """
        Prepare the system for execution by loading any provided context and
        generating the initial task plan. Returns an enhanced prompt ready for
        the first ReAct iteration.
        """
        text_context, question, _ = self._parse_user_input_simple(user_input)
        doc_id = str(uuid.uuid4())

        if text_context:
            logger.info("Loading text context into memory before processing question...")
            self._load_context_only(text_context)

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

    def _create_node(self, structure_output: StructureOutput) -> QueryGraphNode:
        """
        Transform structured agent output into a graph node with computed embedding.
        Combines multiple text fields to create a rich semantic representation.
        """
        node_id = str(uuid.uuid4())
        timestamp = time.time()
        summary = structure_output.summary
        context = structure_output.context
        keywords = structure_output.keywords
        embedding_parts = [
            summary,
            context,
            " ".join(keywords),
            structure_output.core_information,
            structure_output.structure_summary,
            structure_output.supporting_evidence
        ]
        text = " ".join(part for part in embedding_parts if part).strip()
        embedding = self.embedding_module.compute_embedding(text)

        return QueryGraphNode(
            id=node_id,
            summary=summary,
            context=context,
            keywords=keywords,
            embedding=embedding,
            timestamp=timestamp,
            core_information=structure_output.core_information,
            supporting_evidence=structure_output.supporting_evidence,
            structure_summary=structure_output.structure_summary,
            acquisition_logic=(
                structure_output.acquisition_logic
                if structure_output.acquisition_logic and structure_output.acquisition_logic.upper() != "N/A"
                else None
            ),
            links=[]
        )

    def _parse_user_input_simple(self, user_input: str) -> Tuple[str, str, bool]:
        """
        Identify whether the input is context-only or contains a question.
        Supports both English and Chinese context markers.
        Returns (context_text, question_text, is_context_only_flag).
        """
        user_input = user_input.strip()

        if user_input.lower().startswith("context:") or user_input.startswith("上下文："):
            if user_input.lower().startswith("context:"):
                text_context = user_input[8:].strip()
            else:
                text_context = user_input[4:].strip()
            return text_context, "", True

        return "", user_input, False


    def _should_terminate(self) -> bool:
        """
        Check if the execution loop should exit based on task completion status.
        """
        if not self.insight_doc:
            return False

        return not self.insight_doc.current_task

    def export_memory(self, filepath: str):
        """
        Serialize all memory components to JSON for later restoration.
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
        Restore memory state from a previously exported JSON file.
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
        """Provide a compact summary of the current memory state."""
        return (
            f"MemoryBank("
            f"nodes={self.query_graph.get_node_count()}, "
            f"edges={self.query_graph.get_edge_count()}, "
            f"entries={self.interaction_tree.get_total_entries()})"
        )

    def clear_memory(self) -> None:
        """
        Wipe all in-memory state to start fresh. Use this between unrelated sessions
        or when resetting the system. Does not affect exported files.
        """
        logger.info("Clearing all memory...")

        self.query_graph.clear()
        self.retrieval_module.mark_index_dirty()
        self.interaction_tree.clear()
        self.insight_doc = None

        if hasattr(self.adapter, '_pending_conflicts'):
            self.adapter._pending_conflicts.clear()
        if hasattr(self.adapter, '_merge_depth'):
            self.adapter._merge_depth = 0

        logger.info("Memory cleared successfully")
