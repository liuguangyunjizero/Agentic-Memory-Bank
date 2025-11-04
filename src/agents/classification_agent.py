"""
分类/聚类 Agent

职责：对长上下文按主题进行分类/聚类

参考：规范文档第5.1节
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
    """分类 Agent 输入"""
    context: str  # 长上下文文本（可能超长）
    task_goal: Optional[str] = None  # 可选，辅助分类决策


@dataclass
class Cluster:
    """聚类结果"""
    cluster_id: str  # 聚类ID
    context: str  # 一句话主题描述
    content: str  # 属于该主题的原始文本内容
    keywords: List[str]  # 关键词列表


@dataclass
class ClassificationOutput:
    """分类 Agent 输出"""
    should_cluster: bool  # 是否需要聚类
    clusters: List[Cluster]  # 聚类列表


class ClassificationAgent(BaseAgent):
    """
    分类/聚类 Agent

    处理超长文本时使用分块策略
    """

    def __init__(self, llm_client, window_size: int = 8000, chunk_ratio: float = 0.9):
        """
        初始化分类 Agent

        Args:
            llm_client: LLMClient 实例
            window_size: Agent 窗口大小（token）
            chunk_ratio: 分块比例（留余量）
        """
        super().__init__(llm_client)
        self.window_size = window_size
        self.chunk_ratio = chunk_ratio
        logger.info(f"分类Agent初始化: window_size={window_size}, chunk_ratio={chunk_ratio}")

    @classmethod
    def from_config(cls, llm_client, config) -> "ClassificationAgent":
        """从配置创建Agent"""
        return cls(
            llm_client=llm_client,
            window_size=config.CLASSIFICATION_AGENT_WINDOW,
            chunk_ratio=config.CHUNK_RATIO
        )

    def run(self, input_data: ClassificationInput) -> ClassificationOutput:
        """
        执行分类/聚类

        Args:
            input_data: ClassificationInput 实例

        Returns:
            ClassificationOutput 实例
        """
        # 1. 检查是否超长
        token_count = self.llm_client.count_tokens(input_data.context)

        if token_count <= self.window_size:
            # 不超长，直接调用 LLM
            return self._classify_single_chunk(input_data)
        else:
            # 超长，分次加载
            logger.warning(f"上下文超长 ({token_count} tokens)，启用分块处理")
            return self._classify_multiple_chunks(input_data)

    def _classify_single_chunk(self, input_data: ClassificationInput) -> ClassificationOutput:
        """
        处理单个块

        Args:
            input_data: ClassificationInput 实例

        Returns:
            ClassificationOutput 实例
        """
        prompt = self._build_prompt(input_data.context, input_data.task_goal)

        logger.debug("调用LLM进行分类...")
        response = self._call_llm(prompt)

        return self._parse_response(response)

    def _classify_multiple_chunks(self, input_data: ClassificationInput) -> ClassificationOutput:
        """
        处理多个块（分次加载）

        Args:
            input_data: ClassificationInput 实例

        Returns:
            ClassificationOutput 实例
        """
        chunk_size = int(self.window_size * self.chunk_ratio)
        chunks = self._split_by_boundaries(input_data.context, chunk_size)

        logger.info(f"上下文分为 {len(chunks)} 个块")

        all_clusters = []
        for i, chunk in enumerate(chunks, 1):
            logger.debug(f"处理块 {i}/{len(chunks)}")
            chunk_input = ClassificationInput(
                context=chunk,
                task_goal=input_data.task_goal
            )
            chunk_output = self._classify_single_chunk(chunk_input)
            all_clusters.extend(chunk_output.clusters)

        return ClassificationOutput(should_cluster=True, clusters=all_clusters)

    def _build_prompt(self, context: str, task_goal: Optional[str]) -> str:
        """
        构建 prompt

        Args:
            context: 上下文内容
            task_goal: 任务目标（可选）

        Returns:
            完整 prompt
        """
        return CLASSIFICATION_PROMPT.format(
            task_goal=task_goal or "（无）",
            context=context
        )

    def _parse_response(self, response: str) -> ClassificationOutput:
        """
        解析 LLM 响应

        Args:
            response: LLM 响应字符串

        Returns:
            ClassificationOutput 实例
        """
        try:
            data = self._parse_json_response(response)

            clusters = []
            for i, cluster_data in enumerate(data.get("clusters", []), 1):
                cluster = Cluster(
                    cluster_id=cluster_data.get("cluster_id", f"c{i}"),
                    context=cluster_data.get("context", ""),
                    content=cluster_data.get("content", ""),
                    keywords=cluster_data.get("keywords", [])
                )
                clusters.append(cluster)

            return ClassificationOutput(
                should_cluster=data.get("should_cluster", False),
                clusters=clusters
            )

        except Exception as e:
            logger.error(f"解析分类响应失败: {str(e)}")
            # 返回默认聚类，从input context提取关键词作为fallback
            import re
            # 从原始context中提取中文词组和英文单词作为关键词
            context_preview = input_data.context[:500]
            words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]{3,}', context_preview)
            fallback_keywords = list(set(words[:5])) if words else ["默认分类"]

            return ClassificationOutput(
                should_cluster=False,
                clusters=[Cluster(
                    cluster_id="c1",
                    context="解析失败的默认分类",
                    content=input_data.context[:500],
                    keywords=fallback_keywords
                )]
            )

    def _split_by_boundaries(self, text: str, chunk_size: int) -> List[str]:
        """
        按段落边界切分文本

        Args:
            text: 输入文本
            chunk_size: 块大小（按字符估算，约 1/3 token）

        Returns:
            文本块列表
        """
        # 简单实现：按段落切分
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para)

            if current_size + para_size > chunk_size * 3 and current_chunk:
                # 当前块已满，保存并开始新块
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size

        # 保存最后一块
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        return chunks
