"""
Visit工具

访问网页并提取相关信息，使用Jina Reader API进行智能内容提取。

参考：WebResummer的visit工具实现
"""

import logging
import json
from typing import Dict, Any
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class VisitTool:
    """Visit工具：访问网页并提取相关信息"""

    name = "visit"
    description = "Visit a webpage and extract relevant information based on your goal."

    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL of the webpage to visit"
            },
            "goal": {
                "type": "string",
                "description": "What you want to find on this webpage"
            }
        },
        "required": ["url", "goal"]
    }

    def __init__(self, llm_client=None, jina_api_key: str = None, max_content_length: int = 95000):
        """
        初始化Visit工具

        Args:
            llm_client: LLMClient 实例（用于内容提取）
            jina_api_key: Jina Reader API key（如果提供，使用Jina；否则使用BeautifulSoup）
            max_content_length: 最大内容长度（字符数）
        """
        self.llm_client = llm_client
        self.jina_api_key = jina_api_key
        self.max_content_length = max_content_length

        # 根据是否有Jina API key决定使用哪种方式
        if jina_api_key and jina_api_key != "your-jina-api-key-here":
            self.use_jina = True
            logger.info("VisitTool初始化完成 (使用Jina Reader API)")
        else:
            self.use_jina = False
            logger.info("VisitTool初始化完成 (使用BeautifulSoup + LLM提取)")

    def call(self, params: Dict[str, Any]) -> str:
        """
        执行网页访问

        Args:
            params: {"url": str, "goal": str}

        Returns:
            str: JSON格式的提取内容
        """
        url = params.get("url")
        goal = params.get("goal", "")

        if not url:
            error_msg = "Error: url parameter is required"
            logger.error(error_msg)
            return json.dumps({"error": error_msg}, ensure_ascii=False)

        logger.info(f"访问网页: {url}")
        logger.debug(f"提取目标: {goal}")

        try:
            if self.use_jina:
                content = self._jina_reader(url, goal)
            else:
                content = self._beautifulsoup_reader(url, goal)

            output = {
                "url": url,
                "goal": goal,
                "content": content,
                "content_length": len(content)
            }

            logger.info(f"网页访问完成: {len(content)} 字符")
            return json.dumps(output, ensure_ascii=False, indent=2)

        except Exception as e:
            error_msg = f"Visit failed: {str(e)}"
            logger.error(error_msg)
            return json.dumps({"error": error_msg}, ensure_ascii=False)

    def _jina_reader(self, url: str, goal: str) -> str:
        """
        使用Jina Reader API读取网页

        Args:
            url: 网页URL
            goal: 提取目标

        Returns:
            提取的网页内容（markdown格式）
        """
        logger.debug(f"使用Jina Reader API: {url}")

        max_retries = 3
        timeout = 30

        for attempt in range(max_retries):
            try:
                headers = {
                    "Authorization": f"Bearer {self.jina_api_key}",
                    "X-Return-Format": "markdown"  # 返回markdown格式
                }

                response = requests.get(
                    f"https://r.jina.ai/{url}",
                    headers=headers,
                    timeout=timeout
                )

                if response.status_code == 200:
                    webpage_content = response.text

                    # 截断过长内容
                    if len(webpage_content) > self.max_content_length:
                        webpage_content = webpage_content[:self.max_content_length]
                        logger.warning(f"内容过长，已截断到 {self.max_content_length} 字符")

                    # 使用LLM提取相关内容
                    if self.llm_client and goal:
                        extracted_content = self._extract_with_llm(webpage_content, goal, url)
                        return extracted_content

                    return webpage_content

                else:
                    logger.warning(f"Jina Reader返回错误: {response.status_code}")
                    raise ValueError(f"Jina Reader error: {response.status_code}")

            except Exception as e:
                logger.warning(f"Jina Reader尝试 {attempt+1}/{max_retries} 失败: {str(e)}")
                if attempt == max_retries - 1:
                    # 最后一次失败，回退到BeautifulSoup
                    logger.warning("Jina Reader失败，回退到BeautifulSoup")
                    return self._beautifulsoup_reader(url, goal)

        return "[visit] Failed to read page."

    def _beautifulsoup_reader(self, url: str, goal: str) -> str:
        """
        使用BeautifulSoup读取网页（回退方案）

        Args:
            url: 网页URL
            goal: 提取目标

        Returns:
            提取的网页内容
        """
        logger.debug(f"使用BeautifulSoup: {url}")

        try:
            # 1. 获取网页内容
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            response.encoding = response.apparent_encoding

            # 2. 解析HTML
            soup = BeautifulSoup(response.text, 'html.parser')

            # 移除script和style标签
            for script in soup(["script", "style"]):
                script.decompose()

            # 提取文本
            text = soup.get_text()

            # 清理空行
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)

            # 3. 截断过长内容
            if len(text) > self.max_content_length:
                text = text[:self.max_content_length]
                logger.warning(f"内容过长，已截断到 {self.max_content_length} 字符")

            # 4. 如果有LLM客户端且有goal，使用LLM提取相关内容
            if self.llm_client and goal:
                extracted_content = self._extract_with_llm(text, goal, url)
                return extracted_content

            # 5. 否则返回原始文本
            return text

        except Exception as e:
            logger.error(f"BeautifulSoup读取失败: {str(e)}")
            raise

    def _extract_with_llm(self, full_text: str, goal: str, url: str = "") -> str:
        """
        使用LLM从完整文本中提取相关内容

        Args:
            full_text: 完整网页文本
            goal: 提取目标
            url: 网页URL（用于格式化输出）

        Returns:
            提取的相关内容（JSON格式）
        """
        from src.prompts.agent_prompts import VISIT_EXTRACTION_PROMPT

        prompt = VISIT_EXTRACTION_PROMPT.format(
            goal=goal,
            webpage_content=full_text
        )

        try:
            response = self.llm_client.call(prompt)

            # 尝试解析JSON
            try:
                # 提取JSON部分（如果LLM返回了额外的文本）
                if "```json" in response:
                    response = response.split("```json")[1].split("```")[0]
                elif "```" in response:
                    response = response.split("```")[1].split("```")[0]

                data = json.loads(response.strip())

                # 格式化输出
                url_text = url if url else "this webpage"
                result = f"""The useful information in {url_text} for user goal "{goal}" as follows:

Evidence in page:
{data.get('evidence', 'N/A')}

Summary:
{data.get('summary', 'N/A')}
"""
                return result

            except json.JSONDecodeError:
                # 如果JSON解析失败，直接返回LLM的响应
                logger.warning("LLM返回内容不是有效的JSON，返回原始响应")
                return response

        except Exception as e:
            logger.error(f"LLM提取失败: {str(e)}")
            # 回退到返回截断的原始文本
            return full_text[:5000] + "\n... [内容过长，已截断]"

    def __repr__(self) -> str:
        """返回工具摘要"""
        mode = "Jina Reader API" if self.use_jina else "BeautifulSoup"
        return f"VisitTool(name={self.name}, mode={mode})"
