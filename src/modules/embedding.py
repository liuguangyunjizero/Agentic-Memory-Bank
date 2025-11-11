"""
Embedding 计算模块

使用 SentenceTransformer 进行本地 Embedding 计算：
- 批量计算支持
- 向量归一化（用于余弦相似度）
- 相似度计算
"""

import numpy as np
import logging
from typing import List
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingModule:
    """
    Embedding 计算模块（参考 A-mem）

    使用 SentenceTransformer 进行本地计算，无需 API 调用
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        初始化 Embedding 模块

        Args:
            model_name: SentenceTransformer 模型名称
                       'all-MiniLM-L6-v2' - 快速，维度384
                       'all-mpnet-base-v2' - 精确，维度768
        """
        self.model_name = model_name
        logger.info(f"Loading Embedding model: {model_name}")

        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Embedding model loaded successfully, dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"Failed to load Embedding model: {str(e)}")
            raise

    @classmethod
    def from_config(cls, config) -> "EmbeddingModule":
        """
        从 Config 对象创建 Embedding 模块

        Args:
            config: Config 实例

        Returns:
            EmbeddingModule 实例
        """
        return cls(model_name=config.EMBEDDING_MODEL)

    def compute_embedding(self, text: str) -> np.ndarray:
        """
        计算单个文本的 embedding

        Args:
            text: 输入文本（通常是 summary + context + keywords 的组合）

        Returns:
            归一化的 embedding 向量
        """
        if not text or not text.strip():
            logger.warning("Input text is empty, returning zero vector")
            dim = self.model.get_sentence_embedding_dimension()
            return np.zeros(dim)

        # 计算 embedding
        embedding = self.model.encode([text])[0]

        # 归一化（用于余弦相似度）
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def compute_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        批量计算 embedding（提高效率）

        Args:
            texts: 文本列表

        Returns:
            shape 为 (n, dim) 的矩阵，每行一个归一化向量
        """
        if not texts:
            return np.array([])

        # 批量计算
        embeddings = self.model.encode(texts)

        # 批量归一化
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # 避免除以零
        norms = np.where(norms == 0, 1, norms)
        embeddings = embeddings / norms

        return embeddings

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算余弦相似度

        Args:
            vec1, vec2: 已归一化的向量

        Returns:
            相似度分数 [0, 1]
        """
        similarity = float(np.dot(vec1, vec2))
        # 确保在 [0, 1] 范围内（由于浮点误差可能略微超出）
        return max(0.0, min(1.0, similarity))

    def get_embedding_dimension(self) -> int:
        """
        获取 embedding 维度

        Returns:
            向量维度
        """
        return self.model.get_sentence_embedding_dimension()

    def __repr__(self) -> str:
        """返回模块摘要"""
        return f"EmbeddingModule(model={self.model_name}, dim={self.get_embedding_dimension()})"
