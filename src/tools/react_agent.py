"""
Tool-augmented reasoning agent implementing think-act-observe loops.
Executes tasks using external tools and maintains multi-turn conversation state.
"""

import logging
import json
import time
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class MultiTurnReactAgent:
    """
    Orchestrates iterative reasoning cycles with tool access.
    Runs until answer generation or resource limits are reached.
    """

    def __init__(
        self,
        llm_client,
        tools: Dict[str, Any],
        system_message: str,
        max_iterations: int = 60,
        max_context_tokens: int = 128000,
        temperature: float = 0.6,
        top_p: float = 0.95
    ):
        """
        Configure reasoning loop constraints and sampling parameters.
        Higher iteration limit allows thorough information gathering.
        """
        self.llm_client = llm_client
        self.tools = tools
        self.system_message = system_message
        self.max_iterations = max_iterations
        self.max_context_tokens = max_context_tokens
        self.temperature = temperature
        self.top_p = top_p
        logger.info(
            f"MultiTurnReactAgent initialized successfully: "
            f"tools={list(tools.keys())}, "
            f"max_iterations={max_iterations}, "
            f"temp={temperature}, top_p={top_p}"
        )

    def run(self, question: str) -> Dict[str, Any]:
        """
        Execute reasoning loop with tool access until completion.
        Returns full conversation trace including all interactions.
        Question parameter may contain task and memory sections from adapter.
        """
        logger.info(f"Starting ReAct loop (task length: {len(question)} characters)")

        print("\n" + "=" * 80)
        print("🤖 ReAct Agent - Input")
        print("=" * 80)
        print(question)
        print("=" * 80)

        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": question}
        ]
        iterations_left = self.max_iterations

        while iterations_left > 0:
            iterations_left -= 1

            print("\n💭 Thinking...", flush=True)
            try:
                response = self.llm_client.call(
                    messages,
                    temperature=self.temperature,
                    top_p=self.top_p
                )
            except Exception as e:
                logger.error(f"LLM call failed: {str(e)}")
                return {"messages": messages}

            if '<tool_response>' in response:
                pos = response.find('<tool_response>')
                logger.warning(
                    f"Cleaned unexpected <tool_response> tag at position {pos} "
                    f"(response length: {len(response)}). "
                    f"Stop parameter may not be working correctly with this API."
                )
                logger.debug(f"Context: ...{response[max(0, pos-50):min(len(response), pos+50)]}...")
                response = response[:pos]

            print("\n" + "=" * 80)
            print("🤖 ReAct Agent - Output")
            print("=" * 80)
            print(response.strip())
            print("=" * 80)

            messages.append({"role": "assistant", "content": response.strip()})

            if '<tool_call>' in response and '</tool_call>' in response:
                tool_result = self._handle_tool_call(response)
                messages.append({"role": "user", "content": tool_result})

            elif '<answer>' in response and '</answer>' in response:
                logger.debug("Task completed with answer tag")
                return {"messages": messages}

            token_count = self.llm_client.count_tokens(messages)

            if token_count > self.max_context_tokens:
                logger.warning(f"Token limit exceeded: {token_count} > {self.max_context_tokens}")

                force_answer_msg = (
                    "You have now reached the maximum context length. "
                    "Please provide your final answer immediately using the <answer></answer> format."
                )
                messages.append({"role": "user", "content": force_answer_msg})

                response = self.llm_client.call(messages)
                messages.append({"role": "assistant", "content": response.strip()})

                logger.debug("Task terminated due to token limit")
                return {"messages": messages}

        logger.warning(f"Reached maximum iterations: {self.max_iterations}")
        return {"messages": messages}

    def _handle_tool_call(self, response: str) -> str:
        """
        Parse tool request from LLM output, execute, and return formatted result.
        Wraps results in tool_response tags for LLM consumption.
        Returns error messages on invalid JSON or unknown tool names.
        """
        try:
            tool_call_str = response.split('<tool_call>')[1].split('</tool_call>')[0]
            tool_call = json.loads(tool_call_str.strip())

            tool_name = tool_call.get('name', '')
            tool_args = tool_call.get('arguments', {})

            logger.info(f"Executing tool: {tool_name}, arguments: {tool_args}")

            if tool_name in self.tools:
                result = self.tools[tool_name].call(tool_args)
            else:
                result = f"Error: Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}"
                logger.error(result)

            tool_response = f"<tool_response>{result}</tool_response>"

            print(tool_response)

            return tool_response

        except json.JSONDecodeError as e:
            error_msg = f"Error: Invalid JSON in tool call - {str(e)}"
            logger.error(error_msg)
            tool_response = f"<tool_response>{error_msg}</tool_response>"
            print(tool_response)
            return tool_response
        except Exception as e:
            error_msg = f"Error: Tool call failed - {str(e)}"
            logger.error(error_msg)
            tool_response = f"<tool_response>{error_msg}</tool_response>"
            print(tool_response)
            return tool_response

    def __repr__(self) -> str:
        """Show tool availability and iteration limit."""
        return (
            f"MultiTurnReactAgent("
            f"tools={list(self.tools.keys())}, "
            f"max_iterations={self.max_iterations})"
        )
