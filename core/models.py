"""
Data Models Module

Defines all data structures used in the Agentic Memory Bank:
- InsightDoc related models
- QueryGraph related models
- InteractionTree related models
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Optional, Any
import numpy as np


# ============================================================================
# InsightDoc Models
# ============================================================================

@dataclass
class CompletedTask:
    """
    Represents a completed subtask in the InsightDoc.

    Format: [description] → [result] → [impact]
    """
    description: str
    result: str
    impact: Optional[str] = None
    completed_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'description': self.description,
            'result': self.result,
            'impact': self.impact,
            'completed_at': self.completed_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'CompletedTask':
        """Create from dictionary."""
        return cls(
            description=data['description'],
            result=data['result'],
            impact=data.get('impact'),
            completed_at=datetime.fromisoformat(data['completed_at'])
        )

    def format(self) -> str:
        """Format for display in InsightDoc."""
        if self.impact:
            return f"{self.description} → {self.result} → {self.impact}"
        return f"{self.description} → {self.result}"


# ============================================================================
# QueryGraph Models
# ============================================================================

@dataclass
class ChangeLogEntry:
    """
    Records a change to a QueryGraph node.

    Used for tracking small modifications to node content.
    """
    timestamp: datetime
    changed_field: str
    old_value: str
    new_value: str
    reason: str
    source_interaction: str

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'changed_field': self.changed_field,
            'old_value': self.old_value,
            'new_value': self.new_value,
            'reason': self.reason,
            'source_interaction': self.source_interaction
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ChangeLogEntry':
        """Create from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data['timestamp']),
            changed_field=data['changed_field'],
            old_value=data['old_value'],
            new_value=data['new_value'],
            reason=data['reason'],
            source_interaction=data['source_interaction']
        )


@dataclass
class NodeMetadata:
    """
    Metadata for QueryGraph nodes.

    Tracks node status, version relationships, change history, and conflicts.
    """
    status: str = "active"  # "active" | "superseded" | "invalid" | "merged"
    superseded_by: Optional[str] = None
    supersedes: Optional[str] = None
    merged_into: Optional[str] = None
    merged_from: List[str] = field(default_factory=list)
    change_log: List[ChangeLogEntry] = field(default_factory=list)
    conflict_status: str = "none"  # "none" | "unresolved" | "resolved"
    source_type: str = "user_input"  # "user_input" | "search" | "reasoning"
    access_count: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'status': self.status,
            'superseded_by': self.superseded_by,
            'supersedes': self.supersedes,
            'merged_into': self.merged_into,
            'merged_from': self.merged_from,
            'change_log': [entry.to_dict() for entry in self.change_log],
            'conflict_status': self.conflict_status,
            'source_type': self.source_type,
            'access_count': self.access_count
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'NodeMetadata':
        """Create from dictionary."""
        return cls(
            status=data.get('status', 'active'),
            superseded_by=data.get('superseded_by'),
            supersedes=data.get('supersedes'),
            merged_into=data.get('merged_into'),
            merged_from=data.get('merged_from', []),
            change_log=[ChangeLogEntry.from_dict(e) for e in data.get('change_log', [])],
            conflict_status=data.get('conflict_status', 'none'),
            source_type=data.get('source_type', 'user_input'),
            access_count=data.get('access_count', 0)
        )


@dataclass
class QueryGraphNode:
    """
    A node in the Query Graph representing structured memory.

    Contains summary, keywords, tags, embedding, and references to
    Interaction Tree nodes.
    """
    id: str
    summary: str
    keywords: List[str]
    tags: List[str]
    embedding: np.ndarray
    created_at: datetime
    updated_at: datetime
    interaction_refs: Dict[str, List[str]]  # {"text": [...], "webpage": [...], ...}
    metadata: NodeMetadata

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        from utils.serialization import numpy_to_list

        return {
            'id': self.id,
            'summary': self.summary,
            'keywords': self.keywords,
            'tags': self.tags,
            'embedding': numpy_to_list(self.embedding),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'interaction_refs': self.interaction_refs,
            'metadata': self.metadata.to_dict()
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'QueryGraphNode':
        """Create from dictionary."""
        from utils.serialization import list_to_numpy

        return cls(
            id=data['id'],
            summary=data['summary'],
            keywords=data['keywords'],
            tags=data['tags'],
            embedding=list_to_numpy(data['embedding']),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            interaction_refs=data['interaction_refs'],
            metadata=NodeMetadata.from_dict(data['metadata'])
        )


@dataclass
class Edge:
    """
    An edge in the Query Graph representing relationships between nodes.

    Types:
    - "related": Semantic/thematic relationship
    - "conflict": Contradictory information
    """
    from_id: str
    to_id: str
    edge_type: str  # "related" | "conflict"
    created_at: datetime
    created_by: str = "memory_analysis_agent"  # "memory_analysis_agent" | "user"
    note: Optional[str] = None  # Optional explanation

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'from_id': self.from_id,
            'to_id': self.to_id,
            'edge_type': self.edge_type,
            'created_at': self.created_at.isoformat(),
            'created_by': self.created_by,
            'note': self.note
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Edge':
        """Create from dictionary."""
        return cls(
            from_id=data['from_id'],
            to_id=data['to_id'],
            edge_type=data['edge_type'],
            created_at=datetime.fromisoformat(data['created_at']),
            created_by=data.get('created_by', 'memory_analysis_agent'),
            note=data.get('note')
        )


# ============================================================================
# InteractionTree Models
# ============================================================================

@dataclass
class InteractionTreeNode:
    """
    A node in the Interaction Tree storing detailed interaction history.

    Supports multiple modalities:
    - root: Root node with no content
    - text: Text-based interactions
    - webpage: Web search results
    - image: Images and visual content
    - code: Code snippets and execution results
    """
    id: str
    parent_id: Optional[str]
    query_node_ref: str  # Reference to Query Graph node
    modality: str  # "root" | "text" | "webpage" | "image" | "code"
    content: Optional[Dict[str, Any]]  # Content structure varies by modality
    children: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'parent_id': self.parent_id,
            'query_node_ref': self.query_node_ref,
            'modality': self.modality,
            'content': self.content,
            'children': self.children,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'InteractionTreeNode':
        """Create from dictionary."""
        return cls(
            id=data['id'],
            parent_id=data.get('parent_id'),
            query_node_ref=data['query_node_ref'],
            modality=data['modality'],
            content=data.get('content'),
            children=data.get('children', []),
            created_at=datetime.fromisoformat(data['created_at']),
            metadata=data.get('metadata', {})
        )


# ============================================================================
# Export all models
# ============================================================================

__all__ = [
    'CompletedTask',
    'ChangeLogEntry',
    'NodeMetadata',
    'QueryGraphNode',
    'Edge',
    'InteractionTreeNode',
]
