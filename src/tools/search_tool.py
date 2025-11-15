"""
Web search tool using Serper API to query Google and return ranked results.
Supports parallel execution of multiple queries for efficiency.
"""

import logging
import json
import time
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
import requests

logger = logging.getLogger(__name__)


class SearchTool:
    """
    Executes web searches and structures results for agent consumption.
    Handles retries and parallel query processing.
    """

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
        Configure search tool with API credentials and result limits.
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
        Execute one or more search queries and aggregate results.
        Returns JSON containing titles, URLs, and snippets from search results.
        """
        queries = params.get("query", [])

        if not queries:
            error_msg = "Error: query parameter is required and must be a non-empty list"
            logger.error(error_msg)
            return json.dumps({"error": error_msg}, ensure_ascii=False)

        if not isinstance(queries, list):
            queries = [queries]

        logger.info(f"Executing search: {len(queries)} queries")

        try:
            if len(queries) == 1:
                all_results = [self._serper_search_with_retry(queries[0])]
            else:
                with ThreadPoolExecutor(max_workers=3) as executor:
                    all_results = list(executor.map(self._serper_search_with_retry, queries))

            flattened_results = [item for sublist in all_results for item in sublist]

            output = {
                "queries": queries,
                "total_results": len(flattened_results),
                "results": flattened_results
            }

            logger.info(f"Search completed: {len(flattened_results)} results from {len(queries)} queries")
            return json.dumps(output, ensure_ascii=False, indent=2)

        except Exception as e:
            error_msg = f"Search failed: {str(e)}"
            logger.error(error_msg)
            return json.dumps({"error": error_msg}, ensure_ascii=False)

    def _serper_search_with_retry(self, query: str) -> List[Dict[str, str]]:
        """
        Attempt search with exponential backoff on failure.
        Returns empty list rather than raising exception to handle partial failures gracefully.
        """
        max_retries = 5
        for attempt in range(max_retries):
            try:
                return self._serper_search(query)
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Search '{query}' failed after {max_retries} retries: {str(e)}")
                    return []
                logger.warning(f"Search '{query}' attempt {attempt + 1}/{max_retries} failed, retrying...")
                time.sleep(0.5)

    def _serper_search(self, query: str) -> List[Dict[str, str]]:
        """
        Make single HTTP request to Serper API and parse organic search results.
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
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

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
        """Identify tool and API backend."""
        return f"SearchTool(name={self.name}, api=Serper)"
