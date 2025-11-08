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
    task_goal: Optional[str] = None  # 总任务目标（参考）
    current_task: Optional[str] = None  # 当前子任务（参考，帮助识别重要信息）


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

    def __init__(self, llm_client, window_size: int = 8000, chunk_ratio: float = 0.9,
                 temperature: float = 0.4, top_p: float = 0.9):
        """
        初始化分类 Agent

        Args:
            llm_client: LLMClient 实例
            window_size: Agent 窗口大小（token）
            chunk_ratio: 分块比例（留余量）
            temperature: 温度参数
            top_p: 采样参数
        """
        super().__init__(llm_client)
        self.window_size = window_size
        self.chunk_ratio = chunk_ratio
        self.temperature = temperature
        self.top_p = top_p
        logger.info(f"分类Agent初始化: window_size={window_size}, chunk_ratio={chunk_ratio}, "
                   f"temp={temperature}, top_p={top_p}")

    @classmethod
    def from_config(cls, llm_client, config) -> "ClassificationAgent":
        """从配置创建Agent"""
        return cls(
            llm_client=llm_client,
            window_size=config.CLASSIFICATION_AGENT_WINDOW,
            chunk_ratio=config.CHUNK_RATIO,
            temperature=config.CLASSIFICATION_AGENT_TEMPERATURE,
            top_p=config.CLASSIFICATION_AGENT_TOP_P
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
        prompt = self._build_prompt(input_data.context, input_data.task_goal, input_data.current_task)

        logger.debug(f"调用LLM进行分类 (temp={self.temperature}, top_p={self.top_p})...")
        response = self.llm_client.call(prompt, temperature=self.temperature, top_p=self.top_p)

        # 记录LLM原始响应
        logger.debug("="*80)
        logger.debug("📤 Classification Agent LLM原始响应:")
        logger.debug(response)
        logger.debug("="*80)

        return self._parse_response(response, input_data)

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

    def _build_prompt(self, context: str, task_goal: Optional[str], current_task: Optional[str]) -> str:
        """
        构建 prompt

        Args:
            context: 上下文内容
            task_goal: 总任务目标（可选）
            current_task: 当前子任务（可选）

        Returns:
            完整 prompt
        """
        return CLASSIFICATION_PROMPT.format(
            task_goal=task_goal or "（无）",
            current_task=current_task or "（无）",
            context=context
        )

    def _parse_response(self, response: str, input_data: ClassificationInput = None) -> ClassificationOutput:
        """
        解析 LLM 响应（简单分隔符格式，非JSON）

        Args:
            response: LLM 响应字符串
            input_data: ClassificationInput 实例（用于填充content）

        Returns:
            ClassificationOutput 实例
        """
        try:
            # 解析 SHOULD_CLUSTER
            should_cluster = False
            if "SHOULD_CLUSTER:" in response:
                should_cluster_line = [line for line in response.split('\n') if 'SHOULD_CLUSTER:' in line][0]
                should_cluster = 'true' in should_cluster_line.lower()

            # 按 === CLUSTER 分隔符拆分
            cluster_blocks = response.split('=== CLUSTER')[1:]  # 跳过第一部分（SHOULD_CLUSTER行）

            clusters = []
            for i, block in enumerate(cluster_blocks, 1):
                try:
                    # 提取cluster_id（从 "c1 ===" 或 "c2 ===" 中提取）
                    cluster_id_match = block.split('===')[0].strip()
                    cluster_id = cluster_id_match if cluster_id_match else f"c{i}"

                    # 提取 CONTEXT
                    context = ""
                    if "CONTEXT:" in block:
                        context_line = [line for line in block.split('\n') if line.strip().startswith('CONTEXT:')][0]
                        context = context_line.split('CONTEXT:', 1)[1].strip()

                    # 提取 KEYWORDS
                    keywords = []
                    if "KEYWORDS:" in block:
                        keywords_line = [line for line in block.split('\n') if line.strip().startswith('KEYWORDS:')][0]
                        keywords_str = keywords_line.split('KEYWORDS:', 1)[1].strip()
                        keywords = [kw.strip() for kw in keywords_str.split(',')]

                    # ✅ 修复：直接使用原始输入作为content，而不是让LLM复制
                    # 避免浪费token和LLM可能的复制错误
                    content = input_data.context if input_data else ""

                    if not content:
                        logger.warning(f"cluster {i} 缺少content（input_data为空），使用空字符串")

                    # 创建 Cluster 对象
                    cluster = Cluster(
                        cluster_id=cluster_id,
                        context=context,
                        content=content,
                        keywords=keywords
                    )
                    clusters.append(cluster)

                except Exception as e:
                    logger.warning(f"解析cluster块 {i} 失败: {str(e)}, 跳过该块")
                    continue

            # 如果成功解析到cluster，返回结果
            if clusters:
                return ClassificationOutput(
                    should_cluster=should_cluster,
                    clusters=clusters
                )
            else:
                raise ValueError("未能解析出任何cluster")

        except Exception as e:
            logger.error(f"解析分类响应失败: {str(e)}")
            logger.debug(f"响应内容（前1000字符）: {response[:1000]}")

            # Fallback: 返回默认单一cluster
            import re

            # 如果有input_data，从原始context中提取关键词
            if input_data and input_data.context:
                context_preview = input_data.context[:100] + "..." if len(input_data.context) > 100 else input_data.context
                content_fallback = input_data.context
            else:
                context_preview = "解析失败的默认分类"
                content_fallback = response[:500]

            # 从原始context中提取中文词组和英文单词作为关键词
            words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]{3,}', context_preview)
            fallback_keywords = list(set(words[:5])) if words else ["默认分类"]

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
