"""
Search工具

在网络上搜索信息，返回结构化的搜索结果（使用Serper API）。

参考：WebResummer的search工具实现
"""

import logging
import json
from typing import Dict, Any, List
import requests

logger = logging.getLogger(__name__)


class SearchTool:
    """Search工具：在网络上搜索信息"""

    name = "search"
    description = "Search for information on the web. Can search multiple queries at once."

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of search queries (can be one or multiple)"
            }
        },
        "required": ["query"]
    }

    def __init__(self, search_api_key: str, max_results_per_query: int = 5):
        """
        初始化Search工具

        Args:
            search_api_key: 搜索API密钥（Serper API）
            max_results_per_query: 每个查询返回的最大结果数
        """
        if not search_api_key or search_api_key == "your-serper-api-key-here":
            raise ValueError(
                "Serper API key not configured. Please set SERPER_API_KEY in .env file.\n"
                "Register at: https://serper.dev/"
            )

        self.search_api_key = search_api_key
        self.max_results_per_query = max_results_per_query
        logger.info("SearchTool initialized successfully (Serper API)")

    def call(self, params: Dict[str, Any]) -> str:
        """
        执行搜索

        Args:
            params: {"query": [str, str, ...]}

        Returns:
            str: JSON格式的搜索结果
        """
        queries = params.get("query", [])

        if not queries:
            error_msg = "Error: query parameter is required and must be a non-empty list"
            logger.error(error_msg)
            return json.dumps({"error": error_msg}, ensure_ascii=False)

        if not isinstance(queries, list):
            queries = [queries]  # 如果是单个字符串，转换为列表

        logger.info(f"Executing search: {len(queries)} queries")

        try:
            all_results = []

            for query in queries:
                results = self._serper_search(query)
                all_results.extend(results)

            output = {
                "queries": queries,
                "total_results": len(all_results),
                "results": all_results
            }

            logger.info(f"Search completed: {len(all_results)} results")
            return json.dumps(output, ensure_ascii=False, indent=2)

        except Exception as e:
            error_msg = f"Search failed: {str(e)}"
            logger.error(error_msg)
            return json.dumps({"error": error_msg}, ensure_ascii=False)

    def _serper_search(self, query: str) -> List[Dict[str, str]]:
        """
        使用Serper API进行搜索

        Args:
            query: 搜索查询

        Returns:
            搜索结果列表
        """
        try:
            response = requests.post(
                "https://google.serper.dev/search",
                headers={
                    "X-API-KEY": self.search_api_key,
                    "Content-Type": "application/json"
                },
                json={
                    "q": query,
                    "num": self.max_results_per_query
                },
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            # 解析结果
            results = []
            if "organic" in data:
                for item in data["organic"][:self.max_results_per_query]:
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                        "snippet": item.get("snippet", "")
                    })

            if not results:
                logger.warning(f"Search '{query}' returned no results")

            return results

        except requests.exceptions.RequestException as e:
            logger.error(f"Search failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Exception occurred during search: {str(e)}")
            raise

    def __repr__(self) -> str:
        """返回工具摘要"""
        return f"SearchTool(name={self.name}, api=Serper)"
