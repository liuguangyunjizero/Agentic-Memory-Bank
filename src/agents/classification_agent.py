"""
Classification/Clustering Agent

Responsibility: Classify/cluster long context by topics
"""

import json
import logging
from typing import Optional, List
from dataclasses import dataclass
from src.agents.base_agent import BaseAgent
from src.prompts.agent_prompts import CLASSIFICATION_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class ClassificationInput:
    """Classification Agent input"""
    context: str  # Long context text (may be very long)
    task_goal: Optional[str] = None  # Overall task goal (for reference)
    current_task: Optional[str] = None  # Current subtask (for reference, helps identify important info)


@dataclass
class Cluster:
    """Clustering result"""
    cluster_id: str  # Cluster ID
    context: str  # One-sentence topic description
    content: str  # Original text content belonging to this topic
    keywords: List[str]  # Keyword list


@dataclass
class ClassificationOutput:
    """Classification Agent output"""
    should_cluster: bool  # Whether clustering is needed
    clusters: List[Cluster]  # Cluster list


class ClassificationAgent(BaseAgent):
    """
    Classification/Clustering Agent

    Uses chunking strategy when processing very long text
    """

    def __init__(self, llm_client, window_size: int = 32000, chunk_ratio: float = 0.9,
                 temperature: float = 0.4, top_p: float = 0.9):
        """
        Initialize Classification Agent

        Args:
            llm_client: LLMClient instance
            window_size: Agent window size (tokens)
            chunk_ratio: Chunking ratio (leave margin)
            temperature: Temperature parameter
            top_p: Sampling parameter
        """
        super().__init__(llm_client)
        self.window_size = window_size
        self.chunk_ratio = chunk_ratio
        self.temperature = temperature
        self.top_p = top_p
        logger.info(f"Classification Agent initialized: window_size={window_size}, chunk_ratio={chunk_ratio}, "
                   f"temp={temperature}, top_p={top_p}")

    @classmethod
    def from_config(cls, llm_client, config) -> "ClassificationAgent":
        """Create Agent from config"""
        return cls(
            llm_client=llm_client,
            window_size=config.CLASSIFICATION_AGENT_WINDOW,
            chunk_ratio=config.CHUNK_RATIO,
            temperature=config.CLASSIFICATION_AGENT_TEMPERATURE,
            top_p=config.CLASSIFICATION_AGENT_TOP_P
        )

    def run(self, input_data: ClassificationInput) -> ClassificationOutput:
        """
        Execute classification/clustering

        Args:
            input_data: ClassificationInput instance

        Returns:
            ClassificationOutput instance
        """
        # 1. Check if too long
        token_count = self.llm_client.count_tokens(input_data.context)

        if token_count <= self.window_size:
            # Not too long, call LLM directly
            return self._classify_single_chunk(input_data)
        else:
            # Too long, load in chunks
            logger.warning(f"Context too long ({token_count} tokens), enabling chunked processing")
            return self._classify_multiple_chunks(input_data)

    def _classify_single_chunk(self, input_data: ClassificationInput) -> ClassificationOutput:
        """
        Process single chunk

        Args:
            input_data: ClassificationInput instance

        Returns:
            ClassificationOutput instance
        """
        prompt = self._build_prompt(input_data.context, input_data.task_goal, input_data.current_task)

        # Log LLM input
        logger.debug("="*80)
        logger.debug("Classification Agent LLM input:")
        logger.debug(prompt)
        logger.debug("="*80)

        response = self.llm_client.call(prompt, temperature=self.temperature, top_p=self.top_p, stop=None)

        # Log LLM raw response
        logger.debug("="*80)
        logger.debug("Classification Agent LLM raw response:")
        logger.debug(response)
        logger.debug("="*80)

        return self._parse_response(response, input_data)

    def _classify_multiple_chunks(self, input_data: ClassificationInput) -> ClassificationOutput:
        """
        Process multiple chunks (load in batches)

        Args:
            input_data: ClassificationInput instance

        Returns:
            ClassificationOutput instance
        """
        chunk_size = int(self.window_size * self.chunk_ratio)
        chunks = self._split_by_boundaries(input_data.context, chunk_size)

        logger.info(f"Context divided into {len(chunks)} chunks")

        all_clusters = []
        for i, chunk in enumerate(chunks, 1):
            chunk_input = ClassificationInput(
                context=chunk,
                task_goal=input_data.task_goal
            )
            chunk_output = self._classify_single_chunk(chunk_input)
            all_clusters.extend(chunk_output.clusters)

        return ClassificationOutput(should_cluster=True, clusters=all_clusters)

    def _build_prompt(self, context: str, task_goal: Optional[str], current_task: Optional[str]) -> str:
        """
        Build prompt

        Args:
            context: Context content
            task_goal: Overall task goal (optional)
            current_task: Current subtask (optional)

        Returns:
            Complete prompt
        """
        return CLASSIFICATION_PROMPT.format(
            task_goal=task_goal or "(none)",
            current_task=current_task or "(none)",
            context=context
        )

    def _extract_content(self, content_str: str) -> str:
        """
        Extract content between CONTENT_START and CONTENT_END

        Args:
            content_str: String containing CONTENT_START/CONTENT_END markers

        Returns:
            Extracted content, or empty string if extraction fails
        """
        import re
        match = re.search(r'CONTENT_START(.*?)CONTENT_END', content_str, re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            return extracted if extracted else ""
        return ""

    def _parse_response(self, response: str, input_data: ClassificationInput = None) -> ClassificationOutput:
        """
        Parse LLM response (simple delimiter format, not JSON)

        Args:
            response: LLM response string
            input_data: ClassificationInput instance (used to populate content)

        Returns:
            ClassificationOutput instance
        """
        try:
            # Parse SHOULD_CLUSTER
            should_cluster = False
            if "SHOULD_CLUSTER:" in response:
                should_cluster_line = [line for line in response.split('\n') if 'SHOULD_CLUSTER:' in line][0]
                should_cluster = 'true' in should_cluster_line.lower()

            # Split by === CLUSTER delimiter
            cluster_blocks = response.split('=== CLUSTER')[1:]  # Skip first part (SHOULD_CLUSTER line)

            clusters = []
            for i, block in enumerate(cluster_blocks, 1):
                try:
                    # Extract cluster_id (from "c1 ===" or "c2 ===")
                    cluster_id_match = block.split('===')[0].strip()
                    cluster_id = cluster_id_match if cluster_id_match else f"c{i}"

                    # Extract CONTEXT
                    context = ""
                    if "CONTEXT:" in block:
                        context_line = [line for line in block.split('\n') if line.strip().startswith('CONTEXT:')][0]
                        context = context_line.split('CONTEXT:', 1)[1].strip()

                    # Extract KEYWORDS
                    keywords = []
                    if "KEYWORDS:" in block:
                        keywords_line = [line for line in block.split('\n') if line.strip().startswith('KEYWORDS:')][0]
                        keywords_str = keywords_line.split('KEYWORDS:', 1)[1].strip()
                        keywords = [kw.strip() for kw in keywords_str.split(',')]

                    # Extract CONTENT (between CONTENT_START/CONTENT_END)
                    content_from_llm = ""
                    if "CONTENT_START" in block and "CONTENT_END" in block:
                        content_from_llm = self._extract_content(block)

                    # Create Cluster object (use LLM-returned content first, adjust later based on classification)
                    cluster = Cluster(
                        cluster_id=cluster_id,
                        context=context,
                        content=content_from_llm,  # Temporarily use LLM-returned content
                        keywords=keywords
                    )
                    clusters.append(cluster)

                except Exception as e:
                    logger.warning(f"Failed to parse cluster block {i}: {str(e)}, skipping this block")
                    continue

            # If clusters successfully parsed, adjust content based on classification
            if clusters:
                # Determine: if not classifying (only 1 cluster), prefer original context
                # If classifying (multiple clusters), use LLM-split content
                if not should_cluster or len(clusters) == 1:
                    # Not classifying: all clusters use full original context (safest)
                    for cluster in clusters:
                        cluster.content = input_data.context if input_data else cluster.content
                else:
                    # Need classification: validate and fix content
                    for cluster in clusters:
                        if not cluster.content or len(cluster.content.strip()) == 0:
                            # If LLM didn't return content correctly, fallback to original context
                            logger.warning(f"Cluster {cluster.cluster_id} has empty content, using full context as fallback")
                            cluster.content = input_data.context if input_data else ""

                return ClassificationOutput(
                    should_cluster=should_cluster,
                    clusters=clusters
                )
            else:
                raise ValueError("Failed to parse any clusters")

        except Exception as e:
            logger.error(f"Failed to parse classification response: {str(e)}")

            # Fallback: Return default single cluster
            import re

            # If have input_data, extract keywords from full context
            if input_data and input_data.context:
                content_fallback = input_data.context

                # Extract keywords from full context (no truncation)
                words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z0-9]{2,}', input_data.context)

                # Add stopword filtering (common Chinese/English stopwords)
                stopwords = {'the', 'a', 'an', 'for', 'to', 'of', 'in', 'on', 'at', 'is', 'are',
                             'that', 'with', 'from', 'by'}
                filtered_words = [w for w in words if w.lower() not in stopwords]

                # Take first 8 after deduplication (use dict.fromkeys to preserve order)
                fallback_keywords = list(dict.fromkeys(filtered_words))[:8] if filtered_words else ["default_category"]

                # context preview only for display (keep 100-char truncation for context field)
                context_preview = input_data.context[:100] + "..." if len(input_data.context) > 100 else input_data.context
            else:
                context_preview = "Default category for parse failure"
                content_fallback = response[:500]
                fallback_keywords = ["default_category"]

            return ClassificationOutput(
                should_cluster=False,
                clusters=[Cluster(
                    cluster_id="c1",
                    context=context_preview,
                    content=content_fallback,
                    keywords=fallback_keywords
                )]
            )

    def _split_by_boundaries(self, text: str, chunk_size: int) -> List[str]:
        """
        Split text by paragraph boundaries

        Args:
            text: Input text
            chunk_size: Chunk size (by character estimate, about 1/3 token)

        Returns:
            List of text chunks
        """
        # Simple implementation: split by paragraphs
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para)

            if current_size + para_size > chunk_size * 3 and current_chunk:
                # Current chunk is full, save and start new chunk
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size

        # Save last chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        return chunks
