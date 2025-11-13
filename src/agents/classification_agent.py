"""
Classification/Clustering Agent

Responsibility: Classify/cluster long context by topics
"""

import logging
from typing import List
from dataclasses import dataclass
from src.agents.base_agent import BaseAgent
from src.prompts.agent_prompts import CLASSIFICATION_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class ClassificationInput:
    """Classification Agent input"""
    context: str  # Long context text (may be very long)


@dataclass
class Cluster:
    """Clustering result"""
    cluster_id: str  # Cluster ID
    content: str  # Original text chunk (verbatim)


@dataclass
class ClassificationOutput:
    """Classification Agent output"""
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
        prompt = self._build_prompt(input_data.context)

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
                context=chunk
            )
            chunk_output = self._classify_single_chunk(chunk_input)
            all_clusters.extend(chunk_output.clusters)

        return ClassificationOutput(clusters=all_clusters)

    def _build_prompt(self, context: str) -> str:
        """
        Build prompt

        Args:
            context: Context content

        Returns:
            Complete prompt
        """
        return CLASSIFICATION_PROMPT.format(context=context)

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
            # Normalize newlines
            normalized = response.replace('\r', '')

            if "SHOULD_CLUSTER" not in normalized:
                raise ValueError("Missing SHOULD_CLUSTER marker")

            cluster_blocks = [block for block in normalized.split('===') if 'CHUNK' in block]

            clusters = []
            for i, block in enumerate(cluster_blocks, 1):
                try:
                    header, _, remainder = block.partition('===\n')
                    header = header.strip()
                    cluster_id = header.replace('CHUNK', '').strip() or f"chunk_{i}"

                    if "CONTENT_START" not in remainder or "CONTENT_END" not in remainder:
                        raise ValueError("Missing content delimiters")

                    content = self._extract_content(remainder)

                    if not content.strip():
                        raise ValueError("Empty content chunk")

                    clusters.append(Cluster(cluster_id=cluster_id, content=content))

                except Exception as e:
                    logger.warning(f"Failed to parse cluster block {i}: {str(e)}, skipping this block")
                    continue

            if clusters:
                return ClassificationOutput(clusters=clusters)

            raise ValueError("Failed to parse any clusters")

        except Exception as e:
            logger.error(f"Failed to parse classification response: {str(e)}")

            # Fallback: Return default single cluster
            if input_data and input_data.context:
                content_fallback = input_data.context
            else:
                content_fallback = response[:500]

            return ClassificationOutput(
                clusters=[Cluster(
                    cluster_id="chunk_1",
                    content=content_fallback
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
