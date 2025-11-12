"""
Search Tool

Searches for information on the web and returns structured search results (using Serper API).

Reference: WebResummer's search tool implementation
"""

import logging
import json
import time
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
import requests

logger = logging.getLogger(__name__)


class SearchTool:
    """Search Tool: Search for information on the web"""

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
        Initialize Search Tool

        Args:
            search_api_key: Search API key (Serper API)
            max_results_per_query: Maximum number of results per query
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
        Execute search

        Args:
            params: {"query": [str, str, ...]} or {"query": str}

        Returns:
            str: JSON-formatted search results
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
            # Process multiple queries in parallel
            if len(queries) == 1:
                # Single query: call directly (avoid thread pool overhead)
                all_results = [self._serper_search_with_retry(queries[0])]
            else:
                # Multiple queries: parallel processing (reference code style)
                with ThreadPoolExecutor(max_workers=3) as executor:
                    all_results = list(executor.map(self._serper_search_with_retry, queries))

            # Flatten all results
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
        Search with retry mechanism (reference code style)

        Args:
            query: Search query

        Returns:
            List of search results (returns empty list on failure)
        """
        max_retries = 5
        for attempt in range(max_retries):
            try:
                return self._serper_search(query)
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Search '{query}' failed after {max_retries} retries: {str(e)}")
                    return []  # Return empty list instead of throwing exception
                logger.warning(f"Search '{query}' attempt {attempt + 1}/{max_retries} failed, retrying...")
                time.sleep(0.5)

    def _serper_search(self, query: str) -> List[Dict[str, str]]:
        """
        Search using Serper API

        Args:
            query: Search query

        Returns:
            List of search results
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
                timeout=30  # Increased to 30 seconds
            )
            response.raise_for_status()
            data = response.json()

            # Parse results
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
        """Return tool summary"""
        return f"SearchTool(name={self.name}, api=Serper)"
