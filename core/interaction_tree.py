"""
InteractionTree Module

Stores fine-grained interaction history in an immutable tree structure.
Supports multi-modal information storage.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from core.models import InteractionTreeNode
from utils.id_generator import generate_tree_id


class InteractionTree:
    """
    InteractionTree: Interaction History Layer

    Stores detailed interaction logs in a tree structure:
    - Immutable: Only supports appending new branches
    - Multi-modal: Supports text, webpage, image, code branches
    - Branching by modality: Different content types create different branches

    Design principle:
    - One Query Graph node → One or more Interaction Trees
    - Each tree has a root node
    - Children branch by modality (text, webpage, image, code)
    """

    def __init__(self):
        """Initialize an empty Interaction Tree."""
        self.nodes: Dict[str, InteractionTreeNode] = {}  # tree_id -> node
        self.roots: List[str] = []  # List of root node IDs

    def create_tree_root(self, query_node_id: str) -> str:
        """
        Create a root node for a new interaction tree.

        Args:
            query_node_id: Reference to the Query Graph node

        Returns:
            str: Root node ID
        """
        root_id = generate_tree_id()
        root_node = InteractionTreeNode(
            id=root_id,
            parent_id=None,
            query_node_ref=query_node_id,
            modality="root",
            content=None,
            children=[]
        )
        self.nodes[root_id] = root_node
        self.roots.append(root_id)
        return root_id

    def add_text_branch(
        self,
        parent_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a text branch to the tree.

        Args:
            parent_id: Parent node ID
            text: Text content
            metadata: Optional metadata (e.g., token_count)

        Returns:
            str: New node ID
        """
        parent = self.nodes.get(parent_id)
        if not parent:
            raise ValueError(f"Parent node not found: {parent_id}")

        node_id = generate_tree_id()
        node = InteractionTreeNode(
            id=node_id,
            parent_id=parent_id,
            query_node_ref=parent.query_node_ref,
            modality="text",
            content={"text": text},
            metadata=metadata or {}
        )

        self.nodes[node_id] = node
        parent.children.append(node_id)
        return node_id

    def add_webpage_branch(
        self,
        parent_id: str,
        url: str,
        title: str,
        parsed_text: str,
        links: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a webpage branch to the tree.

        Args:
            parent_id: Parent node ID
            url: Webpage URL
            title: Page title
            parsed_text: Extracted text content
            links: List of links found on the page
            metadata: Optional metadata (e.g., fetch_time)

        Returns:
            str: New node ID
        """
        parent = self.nodes.get(parent_id)
        if not parent:
            raise ValueError(f"Parent node not found: {parent_id}")

        node_id = generate_tree_id()
        node = InteractionTreeNode(
            id=node_id,
            parent_id=parent_id,
            query_node_ref=parent.query_node_ref,
            modality="webpage",
            content={
                "url": url,
                "title": title,
                "parsed_text": parsed_text,
                "links": links or []
            },
            metadata=metadata or {}
        )

        self.nodes[node_id] = node
        parent.children.append(node_id)
        return node_id

    def add_image_branch(
        self,
        parent_id: str,
        image_data: str,
        description: str = "",
        ocr_text: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add an image branch to the tree.

        Args:
            parent_id: Parent node ID
            image_data: Image data (base64 encoded or file path)
            description: Image description
            ocr_text: OCR extracted text
            metadata: Optional metadata (e.g., source)

        Returns:
            str: New node ID
        """
        parent = self.nodes.get(parent_id)
        if not parent:
            raise ValueError(f"Parent node not found: {parent_id}")

        node_id = generate_tree_id()
        node = InteractionTreeNode(
            id=node_id,
            parent_id=parent_id,
            query_node_ref=parent.query_node_ref,
            modality="image",
            content={
                "image_data": image_data,
                "description": description,
                "ocr_text": ocr_text
            },
            metadata=metadata or {}
        )

        self.nodes[node_id] = node
        parent.children.append(node_id)
        return node_id

    def add_code_branch(
        self,
        parent_id: str,
        code: str,
        language: str,
        execution_result: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a code branch to the tree.

        Args:
            parent_id: Parent node ID
            code: Code snippet
            language: Programming language
            execution_result: Execution output (if applicable)
            metadata: Optional metadata

        Returns:
            str: New node ID
        """
        parent = self.nodes.get(parent_id)
        if not parent:
            raise ValueError(f"Parent node not found: {parent_id}")

        node_id = generate_tree_id()
        node = InteractionTreeNode(
            id=node_id,
            parent_id=parent_id,
            query_node_ref=parent.query_node_ref,
            modality="code",
            content={
                "code": code,
                "language": language,
                "execution_result": execution_result
            },
            metadata=metadata or {}
        )

        self.nodes[node_id] = node
        parent.children.append(node_id)
        return node_id

    def get_node(self, tree_id: str) -> Optional[InteractionTreeNode]:
        """
        Get a node by ID.

        Args:
            tree_id: Tree node ID

        Returns:
            InteractionTreeNode or None if not found
        """
        return self.nodes.get(tree_id)

    def get_full_tree(self, root_id: str) -> Dict[str, Any]:
        """
        Get the complete tree structure starting from a root.

        Args:
            root_id: Root node ID

        Returns:
            dict: Tree structure with nested children
        """
        root = self.nodes.get(root_id)
        if not root:
            raise ValueError(f"Root node not found: {root_id}")

        def build_tree(node_id: str) -> Dict[str, Any]:
            node = self.nodes[node_id]
            result = {
                'id': node.id,
                'modality': node.modality,
                'content': node.content,
                'metadata': node.metadata,
                'children': []
            }
            for child_id in node.children:
                result['children'].append(build_tree(child_id))
            return result

        return build_tree(root_id)

    def get_branch_by_modality(
        self,
        root_id: str,
        modality: str
    ) -> List[InteractionTreeNode]:
        """
        Get all nodes of a specific modality under a root.

        Args:
            root_id: Root node ID
            modality: Modality type ("text", "webpage", "image", "code")

        Returns:
            List[InteractionTreeNode]: List of nodes with matching modality
        """
        root = self.nodes.get(root_id)
        if not root:
            raise ValueError(f"Root node not found: {root_id}")

        results = []

        def traverse(node_id: str):
            node = self.nodes[node_id]
            if node.modality == modality:
                results.append(node)
            for child_id in node.children:
                traverse(child_id)

        traverse(root_id)
        return results

    def get_path_to_root(self, tree_id: str) -> List[InteractionTreeNode]:
        """
        Get the path from a node to the root.

        Args:
            tree_id: Tree node ID

        Returns:
            List[InteractionTreeNode]: Path from root to node (root first)
        """
        node = self.nodes.get(tree_id)
        if not node:
            raise ValueError(f"Node not found: {tree_id}")

        path = []
        current = node
        while current:
            path.append(current)
            if current.parent_id:
                current = self.nodes.get(current.parent_id)
            else:
                break

        path.reverse()  # Root first
        return path

    def get_all_leaves(self, root_id: str) -> List[InteractionTreeNode]:
        """
        Get all leaf nodes (nodes with no children) under a root.

        Args:
            root_id: Root node ID

        Returns:
            List[InteractionTreeNode]: List of leaf nodes
        """
        root = self.nodes.get(root_id)
        if not root:
            raise ValueError(f"Root node not found: {root_id}")

        leaves = []

        def traverse(node_id: str):
            node = self.nodes[node_id]
            if not node.children:
                leaves.append(node)
            else:
                for child_id in node.children:
                    traverse(child_id)

        traverse(root_id)
        return leaves

    def delete_tree(self, root_id: str) -> None:
        """
        Delete an entire tree.

        Args:
            root_id: Root node ID
        """
        root = self.nodes.get(root_id)
        if not root:
            raise ValueError(f"Root node not found: {root_id}")

        def delete_recursive(node_id: str):
            node = self.nodes[node_id]
            for child_id in node.children:
                delete_recursive(child_id)
            del self.nodes[node_id]

        delete_recursive(root_id)
        if root_id in self.roots:
            self.roots.remove(root_id)

    def delete_branch(self, branch_root_id: str) -> None:
        """
        Delete a branch (subtree) starting from a node.

        Args:
            branch_root_id: Branch root node ID
        """
        node = self.nodes.get(branch_root_id)
        if not node:
            raise ValueError(f"Node not found: {branch_root_id}")

        # Remove from parent's children list
        if node.parent_id:
            parent = self.nodes.get(node.parent_id)
            if parent and branch_root_id in parent.children:
                parent.children.remove(branch_root_id)

        # Delete all descendants
        def delete_recursive(node_id: str):
            current_node = self.nodes[node_id]
            for child_id in current_node.children:
                delete_recursive(child_id)
            del self.nodes[node_id]

        delete_recursive(branch_root_id)

    def to_dict(self) -> dict:
        """
        Convert to dictionary for serialization.

        Returns:
            dict: Serializable dictionary
        """
        return {
            'nodes': {
                node_id: node.to_dict()
                for node_id, node in self.nodes.items()
            },
            'roots': self.roots
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'InteractionTree':
        """
        Create InteractionTree from dictionary.

        Args:
            data: Dictionary containing tree data

        Returns:
            InteractionTree: Reconstructed tree
        """
        tree = cls()
        tree.nodes = {
            node_id: InteractionTreeNode.from_dict(node_data)
            for node_id, node_data in data['nodes'].items()
        }
        tree.roots = data['roots']
        return tree
