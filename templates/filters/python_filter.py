"""
title: Python Code Execution Pipeline
author: open-webui
version: 1.0.0
license: MIT
description: A pipeline that detects Python code in model outputs, executes it, and appends the results.
requirements:
"""

from typing import List, Union, Generator, Iterator
from pydantic import BaseModel, Field
import subprocess
import re


class Pipeline:
    """
    A pipeline to execute Python code embedded in LLM responses.

    This pipeline inspects the model's output for Python code blocks, executes them,
    and appends the results to the conversation.
    """

    class Valves(BaseModel):
        """
        Configuration for the Python Code Execution Pipeline.

        Attributes:
        -----------
        ENABLE_EXECUTION : bool
            Enable or disable Python code execution.
        """
        ENABLE_EXECUTION: bool = Field(
            default=True,
            description="Enable or disable Python code execution."
        )

    def __init__(self):
        """
        Initializes the Pipeline with default or provided settings.
        """
        self.valves = self.Valves()

    async def on_startup(self):
        """
        Called when the server starts.
        """
        print("Python Code Execution Pipeline started.")

    async def on_shutdown(self):
        """
        Called when the server shuts down.
        """
        print("Python Code Execution Pipeline stopped.")

    def execute_python_code(self, code: str) -> str:
        """
        Executes the provided Python code and returns the output.

        Parameters:
        ----------
        code : str
            Python code to execute.

        Returns:
        -------
        str
            The output of the executed Python code.
        """
        try:
            result = subprocess.run(
                ["python", "-c", code], capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            return f"Error executing code: {e.output.strip()}"

    def inlet(self, body: dict, **kwargs) -> dict:
        """
        Pre-processes all incoming user messages.

        Parameters:
        ----------
        body : dict
            The input body containing user messages.

        Returns:
        -------
        dict
            Optionally modified input body.
        """
        messages = body.get("messages", [])
        if not messages:
            return body

        # Example: Log the user message
        user_message = messages[-1].get("content", "")
        print(f"User Message: {user_message}")

        # Optionally, modify the user message here (e.g., sanitize or add context).
        # messages[-1]["content"] = f"Sanitized: {user_message}"

        return body

    def outlet(self, body: dict, **kwargs) -> dict:
        """
        Inspects all outgoing assistant messages for Python code, executes it, and appends the results.

        Parameters:
        ----------
        body : dict
            The outgoing response body containing model-generated messages.

        Returns:
        -------
        dict
            Modified output body with code execution results, if applicable.
        """
        if not self.valves.ENABLE_EXECUTION:
            return body

        messages = body.get("messages", [])
        if not messages:
            return body

        # Inspect the last assistant message for Python code
        last_message = messages[-1].get("content", "")
        python_code_match = re.search(r"```python\n(.*?)```", last_message, re.DOTALL)

        if python_code_match:
            python_code = python_code_match.group(1)
            execution_result = self.execute_python_code(python_code)

            # Append execution result to the message
            messages.append({
                "role": "assistant",
                "content": f"Executed Python Code:\n```\n{execution_result}\n```"
            })

        return body

# Example Usage
if __name__ == "__main__":
    pipeline_instance = Pipeline()

    # Simulated user message and model response
    model_output = {
        "messages": [
            {"role": "user", "content": "Can you solve this?"},
            {
                "role": "assistant",
                "content": (
                    "Sure! Here's the code:\n"
                    "```python\n"
                    "print('Hello, world!')\n"
                    "```"
                ),
            },
        ]
    }

    # Process the user input (inlet)
    processed_input = pipeline_instance.inlet(model_output)
    print("Processed Input:", processed_input)

    # Process the output through the pipeline (outlet)
    processed_output = pipeline_instance.outlet(processed_input)
    print("Processed Output:", processed_output)
