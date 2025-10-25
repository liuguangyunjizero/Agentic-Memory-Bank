"""
QueryGraph Module

Manages the semantic memory graph with nodes and edges.
Supports attribute-based and embedding-based retrieval.
"""

from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import numpy as np
import networkx as nx

from core.models import QueryGraphNode, Edge, NodeMetadata, ChangeLogEntry
from utils.id_generator import generate_node_id
from utils.embedding import EmbeddingManager


class QueryGraph:
    """
    QueryGraph: Semantic Memory Graph Layer

    Manages:
    - Nodes: Structured memory summaries with embeddings
    - Edges: Relationships between nodes (related, conflict)
    - Operations: CRUD, merge, update, version management
    - Retrieval: Attribute-based, embedding-based, hybrid

    Uses NetworkX for graph structure.
    """

    def __init__(self, embedding_manager: Optional[EmbeddingManager] = None):
        """
        Initialize the Query Graph.

        Args:
            embedding_manager: Optional embedding manager for computing embeddings
        """
        self.nodes: Dict[str, QueryGraphNode] = {}  # node_id -> node
        self.graph = nx.DiGraph()  # Directed graph for edges
        self.embedding_manager = embedding_manager or EmbeddingManager()

    # ========================================================================
    # Node Management: CRUD Operations
    # ========================================================================

    def add_node(
        self,
        summary: str,
        keywords: List[str],
        tags: List[str],
        embedding: np.ndarray,
        interaction_refs: Dict[str, List[str]],
        source_type: str = "user_input",
        node_id: Optional[str] = None
    ) -> str:
        """
        Add a new node to the graph.

        Args:
            summary: Structured summary text
            keywords: List of keywords
            tags: List of tags
            embedding: Embedding vector
            interaction_refs: References to Interaction Tree nodes
            source_type: Source of this memory
            node_id: Optional custom node ID

        Returns:
            str: Node ID
        """
        node_id = node_id or generate_node_id()
        now = datetime.now()

        metadata = NodeMetadata(
            status="active",
            source_type=source_type,
            access_count=0
        )

        node = QueryGraphNode(
            id=node_id,
            summary=summary,
            keywords=keywords,
            tags=tags,
            embedding=embedding,
            created_at=now,
            updated_at=now,
            interaction_refs=interaction_refs,
            metadata=metadata
        )

        self.nodes[node_id] = node
        self.graph.add_node(node_id)
        return node_id

    def get_node(self, node_id: str) -> Optional[QueryGraphNode]:
        """
        Get a node by ID.

        Args:
            node_id: Node ID

        Returns:
            QueryGraphNode or None if not found
        """
        node = self.nodes.get(node_id)
        if node:
            # Increment access count
            node.metadata.access_count += 1
        return node

    def get_nodes_by_ids(self, node_ids: List[str]) -> List[QueryGraphNode]:
        """
        Get multiple nodes by IDs.

        Args:
            node_ids: List of node IDs

        Returns:
            List[QueryGraphNode]: List of nodes (skips not found)
        """
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]

    def delete_node(self, node_id: str, cascade: bool = True) -> None:
        """
        Delete a node from the graph.

        Args:
            node_id: Node ID
            cascade: If True, remove all edges connected to this node
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node not found: {node_id}")

        del self.nodes[node_id]

        if cascade:
            self.graph.remove_node(node_id)

    # ========================================================================
    # Node Update Operations
    # ========================================================================

    def update_node_summary(
        self,
        node_id: str,
        new_summary: str,
        reason: str,
        source_interaction: str
    ) -> None:
        """
        Update node summary (small modification).

        Records change in change_log.

        Args:
            node_id: Node ID
            new_summary: New summary text
            reason: Reason for update
            source_interaction: Interaction tree reference for this update
        """
        node = self.nodes.get(node_id)
        if not node:
            raise ValueError(f"Node not found: {node_id}")

        # Record change
        change_entry = ChangeLogEntry(
            timestamp=datetime.now(),
            changed_field="summary",
            old_value=node.summary,
            new_value=new_summary,
            reason=reason,
            source_interaction=source_interaction
        )
        node.metadata.change_log.append(change_entry)

        # Update summary
        node.summary = new_summary
        node.updated_at = datetime.now()

    def update_node_embedding(self, node_id: str, new_embedding: np.ndarray) -> None:
        """
        Update node embedding.

        Args:
            node_id: Node ID
            new_embedding: New embedding vector
        """
        node = self.nodes.get(node_id)
        if not node:
            raise ValueError(f"Node not found: {node_id}")

        node.embedding = new_embedding
        node.updated_at = datetime.now()

    def add_interaction_ref(
        self,
        node_id: str,
        tree_node_id: str,
        modality: str
    ) -> None:
        """
        Add an interaction tree reference to a node.

        Args:
            node_id: Query Graph node ID
            tree_node_id: Interaction Tree node ID
            modality: Modality type ("text", "webpage", "image", "code")
        """
        node = self.nodes.get(node_id)
        if not node:
            raise ValueError(f"Node not found: {node_id}")

        if modality not in node.interaction_refs:
            node.interaction_refs[modality] = []

        if tree_node_id not in node.interaction_refs[modality]:
            node.interaction_refs[modality].append(tree_node_id)
            node.updated_at = datetime.now()

    def increment_access_count(self, node_id: str) -> None:
        """
        Increment access count for a node.

        Args:
            node_id: Node ID
        """
        node = self.nodes.get(node_id)
        if node:
            node.metadata.access_count += 1

    # ========================================================================
    # Edge Management
    # ========================================================================

    def add_edge(
        self,
        from_id: str,
        to_id: str,
        edge_type: str,
        note: Optional[str] = None,
        created_by: str = "memory_analysis_agent"
    ) -> None:
        """
        Add an edge between two nodes.

        Args:
            from_id: Source node ID
            to_id: Target node ID
            edge_type: "related" or "conflict"
            note: Optional note explaining the relationship
            created_by: Who created this edge
        """
        if from_id not in self.nodes:
            raise ValueError(f"Source node not found: {from_id}")
        if to_id not in self.nodes:
            raise ValueError(f"Target node not found: {to_id}")

        edge = Edge(
            from_id=from_id,
            to_id=to_id,
            edge_type=edge_type,
            created_at=datetime.now(),
            created_by=created_by,
            note=note
        )

        # Store edge data in NetworkX graph
        self.graph.add_edge(
            from_id,
            to_id,
            edge_type=edge_type,
            note=note,
            created_at=edge.created_at,
            created_by=created_by
        )

    def remove_edge(self, from_id: str, to_id: str) -> None:
        """
        Remove an edge between two nodes.

        Args:
            from_id: Source node ID
            to_id: Target node ID
        """
        if self.graph.has_edge(from_id, to_id):
            self.graph.remove_edge(from_id, to_id)

    def get_neighbors(
        self,
        node_id: str,
        edge_type: Optional[str] = None
    ) -> List[str]:
        """
        Get neighboring nodes.

        Args:
            node_id: Node ID
            edge_type: Optional filter by edge type

        Returns:
            List[str]: List of neighbor node IDs
        """
        if node_id not in self.graph:
            return []

        neighbors = []

        # Out-neighbors (outgoing edges)
        for neighbor in self.graph.successors(node_id):
            edge_data = self.graph.edges[node_id, neighbor]
            if edge_type is None or edge_data.get('edge_type') == edge_type:
                neighbors.append(neighbor)

        # In-neighbors (incoming edges)
        for neighbor in self.graph.predecessors(node_id):
            edge_data = self.graph.edges[neighbor, node_id]
            if edge_type is None or edge_data.get('edge_type') == edge_type:
                if neighbor not in neighbors:  # Avoid duplicates
                    neighbors.append(neighbor)

        return neighbors

    def get_edges(self, node_id: str) -> List[Edge]:
        """
        Get all edges connected to a node.

        Args:
            node_id: Node ID

        Returns:
            List[Edge]: List of edges
        """
        if node_id not in self.graph:
            return []

        edges = []

        # Outgoing edges
        for _, to_id, data in self.graph.out_edges(node_id, data=True):
            edge = Edge(
                from_id=node_id,
                to_id=to_id,
                edge_type=data.get('edge_type', 'related'),
                created_at=data.get('created_at', datetime.now()),
                created_by=data.get('created_by', 'unknown'),
                note=data.get('note')
            )
            edges.append(edge)

        # Incoming edges
        for from_id, _, data in self.graph.in_edges(node_id, data=True):
            edge = Edge(
                from_id=from_id,
                to_id=node_id,
                edge_type=data.get('edge_type', 'related'),
                created_at=data.get('created_at', datetime.now()),
                created_by=data.get('created_by', 'unknown'),
                note=data.get('note')
            )
            edges.append(edge)

        return edges

    # ========================================================================
    # Node Relationship Operations
    # ========================================================================

    def merge_nodes(
        self,
        node_ids: List[str],
        keep_node_id: Optional[str] = None
    ) -> str:
        """
        Merge multiple nodes into one.

        Args:
            node_ids: List of node IDs to merge
            keep_node_id: Which node to keep (default: first one)

        Returns:
            str: ID of the kept node
        """
        if len(node_ids) < 2:
            raise ValueError("Need at least 2 nodes to merge")

        # Verify all nodes exist
        for nid in node_ids:
            if nid not in self.nodes:
                raise ValueError(f"Node not found: {nid}")

        # Determine which node to keep
        keep_id = keep_node_id or node_ids[0]
        if keep_id not in node_ids:
            raise ValueError(f"keep_node_id {keep_id} not in node_ids")

        keep_node = self.nodes[keep_id]
        merge_ids = [nid for nid in node_ids if nid != keep_id]

        # Merge content
        all_keywords = set(keep_node.keywords)
        all_tags = set(keep_node.tags)
        all_interaction_refs: Dict[str, List[str]] = {
            k: list(v) for k, v in keep_node.interaction_refs.items()
        }

        for merge_id in merge_ids:
            merge_node = self.nodes[merge_id]

            # Merge keywords and tags
            all_keywords.update(merge_node.keywords)
            all_tags.update(merge_node.tags)

            # Merge interaction refs
            for modality, refs in merge_node.interaction_refs.items():
                if modality not in all_interaction_refs:
                    all_interaction_refs[modality] = []
                all_interaction_refs[modality].extend(refs)

            # Transfer edges
            for neighbor in self.graph.successors(merge_id):
                if neighbor != keep_id and not self.graph.has_edge(keep_id, neighbor):
                    edge_data = self.graph.edges[merge_id, neighbor]
                    self.graph.add_edge(keep_id, neighbor, **edge_data)

            for neighbor in self.graph.predecessors(merge_id):
                if neighbor != keep_id and not self.graph.has_edge(neighbor, keep_id):
                    edge_data = self.graph.edges[neighbor, merge_id]
                    self.graph.add_edge(neighbor, keep_id, **edge_data)

            # Mark as merged
            merge_node.metadata.status = "merged"
            merge_node.metadata.merged_into = keep_id

        # Update kept node
        keep_node.keywords = list(all_keywords)
        keep_node.tags = list(all_tags)
        keep_node.interaction_refs = all_interaction_refs
        keep_node.metadata.merged_from = merge_ids
        keep_node.updated_at = datetime.now()

        # Optionally recompute embedding
        # (For now, we keep the original embedding of the kept node)

        return keep_id

    def create_new_version(
        self,
        old_node_id: str,
        new_summary: str,
        new_keywords: List[str],
        new_tags: List[str],
        new_embedding: np.ndarray,
        new_interaction_refs: Dict[str, List[str]],
        source_type: str = "reasoning"
    ) -> str:
        """
        Create a new version of a node (for major modifications).

        Marks old node as superseded.

        Args:
            old_node_id: ID of node to supersede
            new_summary: New summary
            new_keywords: New keywords
            new_tags: New tags
            new_embedding: New embedding
            new_interaction_refs: New interaction references
            source_type: Source type for new node

        Returns:
            str: New node ID
        """
        old_node = self.nodes.get(old_node_id)
        if not old_node:
            raise ValueError(f"Node not found: {old_node_id}")

        # Create new node
        new_node_id = self.add_node(
            summary=new_summary,
            keywords=new_keywords,
            tags=new_tags,
            embedding=new_embedding,
            interaction_refs=new_interaction_refs,
            source_type=source_type
        )

        # Update metadata
        old_node.metadata.status = "superseded"
        old_node.metadata.superseded_by = new_node_id

        new_node = self.nodes[new_node_id]
        new_node.metadata.supersedes = old_node_id

        # Optionally transfer edges to new node
        # (For now, we keep old edges on old node for history)

        return new_node_id

    # ========================================================================
    # Retrieval: Attribute-Based
    # ========================================================================

    def query_by_keywords(
        self,
        keywords: List[str],
        match_mode: str = "any"
    ) -> List[str]:
        """
        Query nodes by keywords.

        Args:
            keywords: List of keywords to search
            match_mode: "any" or "all"

        Returns:
            List[str]: List of matching node IDs
        """
        results = []
        keywords_lower = [kw.lower() for kw in keywords]

        for node_id, node in self.nodes.items():
            if node.metadata.status != "active":
                continue

            node_keywords_lower = [kw.lower() for kw in node.keywords]

            if match_mode == "any":
                if any(kw in node_keywords_lower for kw in keywords_lower):
                    results.append(node_id)
            else:  # "all"
                if all(kw in node_keywords_lower for kw in keywords_lower):
                    results.append(node_id)

        return results

    def query_by_tags(self, tags: List[str]) -> List[str]:
        """
        Query nodes by tags.

        Args:
            tags: List of tags to search

        Returns:
            List[str]: List of matching node IDs
        """
        results = []
        tags_lower = [tag.lower() for tag in tags]

        for node_id, node in self.nodes.items():
            if node.metadata.status != "active":
                continue

            node_tags_lower = [tag.lower() for tag in node.tags]

            if any(tag in node_tags_lower for tag in tags_lower):
                results.append(node_id)

        return results

    def query_by_time_range(
        self,
        start: datetime,
        end: datetime
    ) -> List[str]:
        """
        Query nodes by creation time range.

        Args:
            start: Start datetime
            end: End datetime

        Returns:
            List[str]: List of matching node IDs
        """
        results = []

        for node_id, node in self.nodes.items():
            if start <= node.created_at <= end:
                results.append(node_id)

        return results

    # ========================================================================
    # Retrieval: Embedding-Based
    # ========================================================================

    def query_by_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0,
        only_active: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Query nodes by embedding similarity.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            threshold: Minimum similarity threshold
            only_active: Only return active nodes

        Returns:
            List[Tuple[str, float]]: List of (node_id, similarity_score)
        """
        candidates = []

        for node_id, node in self.nodes.items():
            if only_active and node.metadata.status != "active":
                continue

            similarity = self.embedding_manager.cosine_similarity(
                query_embedding,
                node.embedding
            )

            if similarity >= threshold:
                candidates.append((node_id, similarity))

        # Sort by similarity (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)

        return candidates[:top_k]

    def query_similar_nodes(
        self,
        node_id: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find nodes similar to a given node.

        Args:
            node_id: Reference node ID
            top_k: Number of top results

        Returns:
            List[Tuple[str, float]]: List of (node_id, similarity_score)
        """
        node = self.nodes.get(node_id)
        if not node:
            raise ValueError(f"Node not found: {node_id}")

        results = self.query_by_embedding(
            query_embedding=node.embedding,
            top_k=top_k + 1,  # +1 because the node itself will be included
            only_active=True
        )

        # Remove the query node itself
        results = [(nid, score) for nid, score in results if nid != node_id]

        return results[:top_k]

    # ========================================================================
    # Retrieval: Hybrid
    # ========================================================================

    def hybrid_query(
        self,
        keywords: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        embedding: Optional[np.ndarray] = None,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Hybrid query combining multiple criteria.

        Args:
            keywords: Keywords to search
            tags: Tags to search
            embedding: Embedding vector for similarity search
            top_k: Number of results
            filters: Additional filters (e.g., {"access_count": ">5"})

        Returns:
            List[str]: List of node IDs
        """
        # Start with all active nodes
        candidates = set()

        # Apply attribute filters
        if keywords:
            keyword_results = self.query_by_keywords(keywords, match_mode="any")
            candidates.update(keyword_results)

        if tags:
            tag_results = self.query_by_tags(tags)
            if candidates:
                candidates &= set(tag_results)  # Intersection
            else:
                candidates.update(tag_results)

        # If no attribute filters, start with all active nodes
        if not candidates and not keywords and not tags:
            candidates = {
                nid for nid, node in self.nodes.items()
                if node.metadata.status == "active"
            }

        # Apply embedding similarity
        if embedding is not None:
            embedding_results = self.query_by_embedding(
                query_embedding=embedding,
                top_k=len(candidates) if candidates else top_k,
                only_active=True
            )
            embedding_node_ids = {nid for nid, _ in embedding_results}

            if candidates:
                candidates &= embedding_node_ids
            else:
                candidates = embedding_node_ids

        # Apply additional filters
        if filters:
            filtered = []
            for node_id in candidates:
                node = self.nodes[node_id]
                if self._apply_filters(node, filters):
                    filtered.append(node_id)
            candidates = set(filtered)

        # Return top_k results
        result_list = list(candidates)[:top_k]
        return result_list

    def _apply_filters(self, node: QueryGraphNode, filters: Dict[str, Any]) -> bool:
        """
        Apply filter conditions to a node.

        Args:
            node: Node to filter
            filters: Filter conditions

        Returns:
            bool: True if node passes all filters
        """
        for key, condition in filters.items():
            if key == "access_count":
                if isinstance(condition, str) and condition.startswith(">"):
                    threshold = int(condition[1:])
                    if node.metadata.access_count <= threshold:
                        return False
            elif key == "status":
                if node.metadata.status != condition:
                    return False

        return True

    # ========================================================================
    # Conflict Management
    # ========================================================================

    def mark_conflict(
        self,
        node_ids: List[str],
        description: str
    ) -> None:
        """
        Mark nodes as conflicting.

        Creates conflict edges between all pairs.

        Args:
            node_ids: List of conflicting node IDs
            description: Description of the conflict
        """
        for i, node_id_1 in enumerate(node_ids):
            for node_id_2 in node_ids[i + 1:]:
                self.add_edge(
                    from_id=node_id_1,
                    to_id=node_id_2,
                    edge_type="conflict",
                    note=description
                )

                # Mark nodes as having unresolved conflict
                self.nodes[node_id_1].metadata.conflict_status = "unresolved"
                self.nodes[node_id_2].metadata.conflict_status = "unresolved"

    def get_conflict_nodes(self) -> List[List[str]]:
        """
        Get groups of conflicting nodes.

        Returns:
            List[List[str]]: List of conflict groups
        """
        conflict_groups = []
        visited = set()

        for node_id in self.nodes.keys():
            if node_id in visited:
                continue

            # Find all nodes connected by conflict edges
            conflict_neighbors = self.get_neighbors(node_id, edge_type="conflict")

            if conflict_neighbors:
                group = [node_id] + conflict_neighbors
                conflict_groups.append(group)
                visited.update(group)

        return conflict_groups

    def resolve_conflict(
        self,
        node_ids: List[str],
        resolution: str
    ) -> None:
        """
        Resolve conflicts between nodes.

        Args:
            node_ids: Conflicting node IDs
            resolution: "keep_first" | "merge" | "invalidate_all"
        """
        if resolution == "keep_first":
            # Keep first node, mark others as invalid
            for node_id in node_ids[1:]:
                self.nodes[node_id].metadata.status = "invalid"
            self.nodes[node_ids[0]].metadata.conflict_status = "resolved"

        elif resolution == "merge":
            # Merge all nodes
            self.merge_nodes(node_ids)

        elif resolution == "invalidate_all":
            # Mark all as invalid
            for node_id in node_ids:
                self.nodes[node_id].metadata.status = "invalid"

        # Remove conflict edges
        for i, node_id_1 in enumerate(node_ids):
            for node_id_2 in node_ids[i + 1:]:
                if self.graph.has_edge(node_id_1, node_id_2):
                    self.remove_edge(node_id_1, node_id_2)
                if self.graph.has_edge(node_id_2, node_id_1):
                    self.remove_edge(node_id_2, node_id_1)

    # ========================================================================
    # Statistics
    # ========================================================================

    def get_graph_stats(self) -> Dict[str, Any]:
        """
        Get graph statistics.

        Returns:
            dict: Statistics including node count, edge count, etc.
        """
        active_nodes = [n for n in self.nodes.values() if n.metadata.status == "active"]

        return {
            "total_nodes": len(self.nodes),
            "active_nodes": len(active_nodes),
            "total_edges": self.graph.number_of_edges(),
            "avg_degree": sum(dict(self.graph.degree()).values()) / len(self.graph.nodes()) if self.graph.nodes() else 0,
            "conflict_groups": len(self.get_conflict_nodes())
        }

    def get_orphan_nodes(self) -> List[str]:
        """
        Get nodes with no edges.

        Returns:
            List[str]: List of orphan node IDs
        """
        orphans = []
        for node_id in self.nodes.keys():
            if self.graph.degree(node_id) == 0:
                orphans.append(node_id)
        return orphans

    # ========================================================================
    # Serialization
    # ========================================================================

    def to_dict(self) -> dict:
        """
        Convert to dictionary for serialization.

        Returns:
            dict: Serializable dictionary
        """
        edges_data = []
        for from_id, to_id, data in self.graph.edges(data=True):
            edge = Edge(
                from_id=from_id,
                to_id=to_id,
                edge_type=data.get('edge_type', 'related'),
                created_at=data.get('created_at', datetime.now()),
                created_by=data.get('created_by', 'unknown'),
                note=data.get('note')
            )
            edges_data.append(edge.to_dict())

        return {
            'nodes': {
                node_id: node.to_dict()
                for node_id, node in self.nodes.items()
            },
            'edges': edges_data
        }

    @classmethod
    def from_dict(
        cls,
        data: dict,
        embedding_manager: Optional[EmbeddingManager] = None
    ) -> 'QueryGraph':
        """
        Create QueryGraph from dictionary.

        Args:
            data: Dictionary containing graph data
            embedding_manager: Optional embedding manager

        Returns:
            QueryGraph: Reconstructed graph
        """
        graph = cls(embedding_manager=embedding_manager)

        # Restore nodes
        graph.nodes = {
            node_id: QueryGraphNode.from_dict(node_data)
            for node_id, node_data in data['nodes'].items()
        }

        # Add nodes to NetworkX graph
        for node_id in graph.nodes.keys():
            graph.graph.add_node(node_id)

        # Restore edges
        for edge_data in data['edges']:
            edge = Edge.from_dict(edge_data)
            graph.graph.add_edge(
                edge.from_id,
                edge.to_id,
                edge_type=edge.edge_type,
                note=edge.note,
                created_at=edge.created_at,
                created_by=edge.created_by
            )

        return graph
