"""
ReAct Agent

Multi-turn conversational Agent supporting Think-Act-Observe loops and tool calling.
"""

import logging
import json
import time
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class MultiTurnReactAgent:
    """
    Multi-turn ReAct Agent

    Features:
    - Think-Act-Observe loop
    - Tool calling (search, visit, deep_retrieval)
    - Stop condition: <answer> tag
    - Context management: Token counting + overflow handling
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
        Initialize ReAct Agent

        Args:
            llm_client: LLMClient instance
            tools: Tool dictionary {tool_name: tool_instance}
            system_message: System Prompt
            max_iterations: Maximum number of iterations
            max_context_tokens: Maximum context token count
            temperature: Temperature parameter
            top_p: Sampling parameter
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
        Execute ReAct loop

        Args:
            question: User question (may have been enhanced by Adapter)

        Returns:
            Execution result: {"messages": List[Dict[str, str]]}
            messages: Complete conversation trace, including all system/user/assistant messages
        """
        logger.info(f"Starting ReAct loop (task length: {len(question)} characters)")

        # Display initial input (task + memory) - only once at start
        print("\n" + "=" * 80)
        print("🤖 ReAct Agent - Input")
        print("=" * 80)
        print(question)
        print("=" * 80)

        # 1. Initialize
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": question}
        ]
        iterations_left = self.max_iterations

        # 2. Main loop
        while iterations_left > 0:
            iterations_left -= 1

            # 2.1 Call LLM
            try:
                response = self.llm_client.call(
                    messages,
                    temperature=self.temperature,
                    top_p=self.top_p
                )
            except Exception as e:
                logger.error(f"LLM call failed: {str(e)}")
                return {"messages": messages}

            # 2.2 Clean unexpected tool_response tags
            # Note: stop parameter should prevent this, but DeepSeek API may not fully honor it
            # This is a safety check to ensure clean responses
            if '<tool_response>' in response:
                pos = response.find('<tool_response>')
                logger.warning(
                    f"Cleaned unexpected <tool_response> tag at position {pos} "
                    f"(response length: {len(response)}). "
                    f"Stop parameter may not be working correctly with this API."
                )
                logger.debug(f"Context: ...{response[max(0, pos-50):min(len(response), pos+50)]}...")
                response = response[:pos]

            # 2.3 Display cleaned LLM output
            print("\n" + "=" * 80)
            print("🤖 ReAct Agent - Output")
            print("=" * 80)
            print(response.strip())
            print("=" * 80)

            # 2.4 Add to message history
            messages.append({"role": "assistant", "content": response.strip()})

            # 2.5 Check for tool calls
            if '<tool_call>' in response and '</tool_call>' in response:
                tool_result = self._handle_tool_call(response)
                messages.append({"role": "user", "content": tool_result})

            # 2.6 Check if completed (answer tag indicates task completion)
            elif '<answer>' in response and '</answer>' in response:
                logger.debug("Task completed with answer tag")
                return {"messages": messages}

            # 2.7 Token counting and context management
            token_count = self.llm_client.count_tokens(messages)

            # 2.8 Overflow handling
            if token_count > self.max_context_tokens:
                logger.warning(f"Token limit exceeded: {token_count} > {self.max_context_tokens}")

                # Force answer generation
                force_answer_msg = (
                    "You have now reached the maximum context length. "
                    "Please provide your final answer immediately using the <answer></answer> format."
                )
                messages.append({"role": "user", "content": force_answer_msg})

                response = self.llm_client.call(messages)
                messages.append({"role": "assistant", "content": response.strip()})

                logger.debug("Task terminated due to token limit")
                return {"messages": messages}

        # 3. Exceeded maximum iterations
        logger.warning(f"Reached maximum iterations: {self.max_iterations}")
        return {"messages": messages}

    def _handle_tool_call(self, response: str) -> str:
        """
        Handle tool calls

        Args:
            response: LLM response (containing <tool_call> tag)

        Returns:
            Tool response (containing <tool_response> tag)
        """
        try:
            # 1. Extract JSON
            tool_call_str = response.split('<tool_call>')[1].split('</tool_call>')[0]
            tool_call = json.loads(tool_call_str.strip())

            # 2. Execute tool
            tool_name = tool_call.get('name', '')
            tool_args = tool_call.get('arguments', {})

            logger.info(f"Executing tool: {tool_name}, arguments: {tool_args}")

            if tool_name in self.tools:
                result = self.tools[tool_name].call(tool_args)
            else:
                result = f"Error: Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}"
                logger.error(result)

            tool_response = f"<tool_response>{result}</tool_response>"

            # Display tool response in console
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
        """Return Agent summary"""
        return (
            f"MultiTurnReactAgent("
            f"tools={list(self.tools.keys())}, "
            f"max_iterations={self.max_iterations})"
        )
