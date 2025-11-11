"""
文件工具模块

提供跨平台的文件操作功能：
- 临时文件管理
- 正式存储管理
- 多模态文件处理（图片、文档等）
- 使用 pathlib 确保跨平台兼容性
"""

import base64
import shutil
import time
import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


class FileUtils:
    """
    文件工具类

    使用 pathlib.Path 确保 Windows/Linux/Mac 跨平台兼容
    """

    def __init__(
        self,
        temp_dir: Union[str, Path] = "data/temp",
        storage_dir: Union[str, Path] = "data/storage"
    ):
        """
        初始化文件工具

        Args:
            temp_dir: 临时文件目录（多模态暂存）
            storage_dir: 正式存储目录
        """
        # 转换为 Path 对象（确保跨平台）
        self.temp_dir = Path(temp_dir)
        self.storage_dir = Path(storage_dir)

        # 确保目录存在
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"File utilities initialized: temp_dir={self.temp_dir}, "
            f"storage_dir={self.storage_dir}"
        )

    @classmethod
    def from_config(cls, config) -> "FileUtils":
        """
        从 Config 对象创建文件工具

        Args:
            config: Config 实例

        Returns:
            FileUtils 实例
        """
        return cls(
            temp_dir=config.TEMP_DIR,
            storage_dir=config.STORAGE_DIR
        )

    def save_to_temp(
        self,
        file_data: bytes,
        temp_id: str,
        file_type: str
    ) -> Path:
        """
        保存文件到临时目录

        Args:
            file_data: 文件二进制数据
            temp_id: 临时文件 ID
            file_type: 文件类型（image/document/code/text）

        Returns:
            保存后的文件路径（Path 对象）
        """
        # 获取文件扩展名
        ext = self._get_extension(file_type)
        filename = f"{temp_id}.{ext}"
        filepath = self.temp_dir / filename

        # 保存文件
        with open(filepath, 'wb') as f:
            f.write(file_data)

        logger.info(f"File saved to temp directory: {filepath}")
        return filepath

    def move_to_storage(
        self,
        temp_path: Union[str, Path],
        node_id: str = None
    ) -> Path:
        """
        将文件从临时目录移动到正式存储目录

        Args:
            temp_path: 临时文件路径
            node_id: 节点 ID（可选，用于生成新文件名）

        Returns:
            正式存储路径（Path 对象）
        """
        temp_path = Path(temp_path)

        # 生成新文件名
        if node_id:
            ext = temp_path.suffix
            filename = f"{node_id}_{int(time.time())}{ext}"
        else:
            filename = temp_path.name

        storage_path = self.storage_dir / filename

        # 移动文件
        shutil.move(str(temp_path), str(storage_path))

        logger.info(f"File moved to storage: {storage_path}")
        return storage_path

    def read_image_as_base64(self, filepath: Union[str, Path]) -> str:
        """
        读取图片并转换为 base64 编码

        Args:
            filepath: 图片文件路径

        Returns:
            Base64 编码的字符串
        """
        filepath = Path(filepath)

        with open(filepath, 'rb') as f:
            image_data = f.read()

        encoded = base64.b64encode(image_data).decode('utf-8')

        return encoded

    def read_text(self, filepath: Union[str, Path], encoding: str = 'utf-8') -> str:
        """
        读取文本文件

        Args:
            filepath: 文本文件路径
            encoding: 文件编码（默认 utf-8）

        Returns:
            文件内容
        """
        filepath = Path(filepath)

        with open(filepath, 'r', encoding=encoding) as f:
            content = f.read()

        return content

    def read_document(self, filepath: Union[str, Path]) -> str:
        """
        读取文档文件（PDF 等）并转换为 base64

        Args:
            filepath: 文档文件路径

        Returns:
            Base64 编码的字符串
        """
        filepath = Path(filepath)

        with open(filepath, 'rb') as f:
            doc_data = f.read()

        encoded = base64.b64encode(doc_data).decode('utf-8')

        return encoded

    def write_text(
        self,
        filepath: Union[str, Path],
        content: str,
        encoding: str = 'utf-8'
    ):
        """
        写入文本文件

        Args:
            filepath: 目标文件路径
            content: 文件内容
            encoding: 文件编码（默认 utf-8）
        """
        filepath = Path(filepath)

        # 确保父目录存在
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding=encoding) as f:
            f.write(content)

    def delete_file(self, filepath: Union[str, Path]):
        """
        删除文件

        Args:
            filepath: 要删除的文件路径
        """
        filepath = Path(filepath)

        if filepath.exists():
            filepath.unlink()
            logger.info(f"File deleted: {filepath}")
        else:
            logger.warning(f"File does not exist, cannot delete: {filepath}")

    def cleanup_temp_dir(self):
        """
        清理临时目录中的所有文件
        """
        if self.temp_dir.exists():
            for file_path in self.temp_dir.iterdir():
                if file_path.is_file():
                    file_path.unlink()

            logger.info("Temp directory cleaned up")

    def _get_extension(self, file_type: str) -> str:
        """
        根据文件类型返回扩展名

        Args:
            file_type: 文件类型

        Returns:
            文件扩展名（不含点）
        """
        type_map = {
            "image": "png",
            "document": "pdf",
            "code": "txt",
            "text": "txt",
            "json": "json"
        }
        return type_map.get(file_type.lower(), "bin")

    def __repr__(self) -> str:
        """返回工具摘要"""
        return (
            f"FileUtils(temp_dir={self.temp_dir}, "
            f"storage_dir={self.storage_dir})"
        )
