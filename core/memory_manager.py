"""
MemoryManager Module

Global manager that integrates all three layers:
- InsightDoc
- QueryGraph
- InteractionTree

Provides high-level APIs for memory operations.
"""

from typing import List, Dict, Optional, Any
from datetime import datetime

from core.insight_doc import InsightDoc
from core.query_graph import QueryGraph
from core.interaction_tree import InteractionTree
from core.models import QueryGraphNode
from utils.embedding import EmbeddingManager
from utils.serialization import save_to_json, load_from_json


class MemoryManager:
    """
    MemoryManager: Global Memory Management

    Integrates:
    - InsightDoc: Task state management
    - QueryGraph: Semantic memory graph
    - InteractionTree: Detailed interaction history
    - EmbeddingManager: Embedding computation

    Provides unified APIs for:
    - Task creation and management
    - Memory addition and retrieval
    - Cross-layer operations
    - Persistence (save/load)
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the Memory Manager.

        Args:
            embedding_model: Name of the sentence-transformers model
        """
        self.embedding_manager = EmbeddingManager(model_name=embedding_model)
        self.query_graph = QueryGraph(embedding_manager=self.embedding_manager)
        self.interaction_tree = InteractionTree()
        self._insight_doc: Optional[InsightDoc] = None

    # ========================================================================
    # Task Management
    # ========================================================================

    def create_task(
        self,
        user_question: str,
        understood_goal: Optional[str] = None
    ) -> str:
        """
        Create a new task.

        Args:
            user_question: User's original question/request
            understood_goal: Planning agent's interpretation (optional)

        Returns:
            str: Task ID
        """
        self._insight_doc = InsightDoc(
            user_question=user_question,
            understood_goal=understood_goal
        )
        return self._insight_doc.task_id

    @property
    def insight_doc(self) -> InsightDoc:
        """
        Get the current InsightDoc.

        Returns:
            InsightDoc: Current task state

        Raises:
            RuntimeError: If no task has been created
        """
        if self._insight_doc is None:
            raise RuntimeError("No task created. Call create_task() first.")
        return self._insight_doc

    @property
    def query_graph(self) -> QueryGraph:
        """Get the Query Graph."""
        return self._query_graph

    @query_graph.setter
    def query_graph(self, value: QueryGraph):
        """Set the Query Graph."""
        self._query_graph = value

    @property
    def interaction_tree(self) -> InteractionTree:
        """Get the Interaction Tree."""
        return self._interaction_tree

    @interaction_tree.setter
    def interaction_tree(self, value: InteractionTree):
        """Set the Interaction Tree."""
        self._interaction_tree = value

    @property
    def embedding_manager(self) -> EmbeddingManager:
        """Get the Embedding Manager."""
        return self._embedding_manager

    @embedding_manager.setter
    def embedding_manager(self, value: EmbeddingManager):
        """Set the Embedding Manager."""
        self._embedding_manager = value

    # ========================================================================
    # Memory Addition (Cross-Layer Operation)
    # ========================================================================

    def add_memory(
        self,
        summary: str,
        keywords: List[str],
        tags: List[str],
        text_content: Optional[str] = None,
        webpage_content: Optional[Dict[str, Any]] = None,
        image_content: Optional[Dict[str, Any]] = None,
        code_content: Optional[Dict[str, Any]] = None,
        source_type: str = "user_input"
    ) -> str:
        """
        Add a new memory across all layers.

        This is a high-level operation that:
        1. Computes embedding for the summary
        2. Creates a Query Graph node
        3. Creates an Interaction Tree with appropriate branches
        4. Links them together

        Args:
            summary: Structured summary text
            keywords: List of keywords
            tags: List of tags
            text_content: Optional text content (for text branch)
            webpage_content: Optional webpage data (dict with url, title, parsed_text, links)
            image_content: Optional image data (dict with image_data, description, ocr_text)
            code_content: Optional code data (dict with code, language, execution_result)
            source_type: Source of this memory

        Returns:
            str: Query Graph node ID
        """
        # 1. Compute embedding
        embedding = self.embedding_manager.compute_embedding(summary)

        # 2. Create Interaction Tree root
        tree_root_id = self.interaction_tree.create_tree_root(query_node_id="temp")

        # 3. Add branches based on content
        interaction_refs: Dict[str, List[str]] = {}

        if text_content:
            text_node_id = self.interaction_tree.add_text_branch(
                parent_id=tree_root_id,
                text=text_content
            )
            interaction_refs["text"] = [text_node_id]

        if webpage_content:
            webpage_node_id = self.interaction_tree.add_webpage_branch(
                parent_id=tree_root_id,
                url=webpage_content.get("url", ""),
                title=webpage_content.get("title", ""),
                parsed_text=webpage_content.get("parsed_text", ""),
                links=webpage_content.get("links", [])
            )
            interaction_refs["webpage"] = [webpage_node_id]

        if image_content:
            image_node_id = self.interaction_tree.add_image_branch(
                parent_id=tree_root_id,
                image_data=image_content.get("image_data", ""),
                description=image_content.get("description", ""),
                ocr_text=image_content.get("ocr_text", "")
            )
            interaction_refs["image"] = [image_node_id]

        if code_content:
            code_node_id = self.interaction_tree.add_code_branch(
                parent_id=tree_root_id,
                code=code_content.get("code", ""),
                language=code_content.get("language", ""),
                execution_result=code_content.get("execution_result")
            )
            interaction_refs["code"] = [code_node_id]

        # 4. Create Query Graph node
        node_id = self.query_graph.add_node(
            summary=summary,
            keywords=keywords,
            tags=tags,
            embedding=embedding,
            interaction_refs=interaction_refs,
            source_type=source_type
        )

        # 5. Update tree root with correct query_node_ref
        tree_root = self.interaction_tree.get_node(tree_root_id)
        if tree_root:
            tree_root.query_node_ref = node_id

        return node_id

    # ========================================================================
    # Memory Retrieval
    # ========================================================================

    def retrieve_memories(
        self,
        query: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        top_k: int = 5,
        only_active: bool = True
    ) -> List[QueryGraphNode]:
        """
        Retrieve memories using hybrid search.

        Args:
            query: Text query (will compute embedding)
            keywords: Keywords to search
            tags: Tags to filter
            top_k: Number of results
            only_active: Only return active nodes

        Returns:
            List[QueryGraphNode]: List of matching nodes
        """
        # Compute query embedding if query text provided
        query_embedding = None
        if query:
            query_embedding = self.embedding_manager.compute_embedding(query)

        # Use hybrid query
        if query_embedding is not None or keywords or tags:
            node_ids = self.query_graph.hybrid_query(
                keywords=keywords,
                tags=tags,
                embedding=query_embedding,
                top_k=top_k,
                filters={"status": "active"} if only_active else None
            )
        else:
            # No criteria, just return recent active nodes
            node_ids = [
                nid for nid, node in self.query_graph.nodes.items()
                if not only_active or node.metadata.status == "active"
            ][:top_k]

        return self.query_graph.get_nodes_by_ids(node_ids)

    def get_full_memory_context(
        self,
        node_id: str,
        modality: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get full memory context including Query Graph node and Interaction Tree.

        Args:
            node_id: Query Graph node ID
            modality: Optional filter by modality

        Returns:
            dict: Complete memory context
        """
        node = self.query_graph.get_node(node_id)
        if not node:
            raise ValueError(f"Node not found: {node_id}")

        context = {
            "node_id": node.id,
            "summary": node.summary,
            "keywords": node.keywords,
            "tags": node.tags,
            "created_at": node.created_at.isoformat(),
            "metadata": node.metadata.to_dict(),
            "interactions": {}
        }

        # Get interaction tree content
        if modality:
            # Get specific modality
            if modality in node.interaction_refs:
                context["interactions"][modality] = []
                for tree_id in node.interaction_refs[modality]:
                    tree_node = self.interaction_tree.get_node(tree_id)
                    if tree_node:
                        context["interactions"][modality].append({
                            "id": tree_node.id,
                            "content": tree_node.content,
                            "metadata": tree_node.metadata
                        })
        else:
            # Get all modalities
            for mod, tree_ids in node.interaction_refs.items():
                context["interactions"][mod] = []
                for tree_id in tree_ids:
                    tree_node = self.interaction_tree.get_node(tree_id)
                    if tree_node:
                        context["interactions"][mod].append({
                            "id": tree_node.id,
                            "content": tree_node.content,
                            "metadata": tree_node.metadata
                        })

        return context

    # ========================================================================
    # Persistence
    # ========================================================================

    def save(self, filepath: str) -> None:
        """
        Save entire memory bank to JSON file.

        Args:
            filepath: Path to save file
        """
        data = {
            "insight_doc": self._insight_doc.to_dict() if self._insight_doc else None,
            "query_graph": self.query_graph.to_dict(),
            "interaction_tree": self.interaction_tree.to_dict(),
            "embedding_model": self.embedding_manager.model_name
        }

        save_to_json(data, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'MemoryManager':
        """
        Load memory bank from JSON file.

        Args:
            filepath: Path to load file

        Returns:
            MemoryManager: Loaded memory manager
        """
        data = load_from_json(filepath)

        # Create manager with same embedding model
        manager = cls(embedding_model=data.get("embedding_model", "all-MiniLM-L6-v2"))

        # Restore InsightDoc
        if data.get("insight_doc"):
            manager._insight_doc = InsightDoc.from_dict(data["insight_doc"])

        # Restore QueryGraph
        manager.query_graph = QueryGraph.from_dict(
            data["query_graph"],
            embedding_manager=manager.embedding_manager
        )

        # Restore InteractionTree
        manager.interaction_tree = InteractionTree.from_dict(data["interaction_tree"])

        return manager

    # ========================================================================
    # Statistics
    # ========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get overall statistics.

        Returns:
            dict: Statistics about the memory bank
        """
        graph_stats = self.query_graph.get_graph_stats()

        stats = {
            "task_id": self._insight_doc.task_id if self._insight_doc else None,
            "completed_tasks": len(self._insight_doc.completed_tasks) if self._insight_doc else 0,
            "pending_tasks": len(self._insight_doc.pending_tasks) if self._insight_doc else 0,
            "total_nodes": graph_stats["total_nodes"],
            "active_nodes": graph_stats["active_nodes"],
            "total_edges": graph_stats["total_edges"],
            "total_tree_nodes": len(self.interaction_tree.nodes),
            "tree_roots": len(self.interaction_tree.roots)
        }

        return stats

    def __str__(self) -> str:
        """String representation for debugging."""
        stats = self.get_statistics()
        return f"MemoryManager(nodes={stats['active_nodes']}, edges={stats['total_edges']}, trees={stats['tree_roots']})"
