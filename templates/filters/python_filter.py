"""
title: Python Code Execution Filter
author: open-webui
version: 1.0.1
license: MIT
description: A filter that detects Python code in model outputs, executes it, and appends the results to the conversation.
requirements: 
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field
import subprocess
import re
from utils.pipelines.main import get_last_user_message, get_last_assistant_message

class Pipeline:
    """
    A filter to execute Python code embedded in LLM responses.

    This filter inspects the model's output for Python code blocks, executes them,
    and appends the results to the conversation.
    """

    class Valves(BaseModel):
        """
        Configuration for the Python Code Execution Filter.

        Attributes:
        -----------
        pipelines : List[str]
            List of pipelines this filter listens to. Use ["*"] to listen to all.
        priority : int
            Priority of the filter in the pipeline.
        ENABLE_EXECUTION : bool
            Enable or disable Python code execution.
        """
        pipelines: List[str] = Field(default=["*"], description="Pipelines to connect to.")
        priority: int = Field(default=0, description="Filter priority.")
        ENABLE_EXECUTION: bool = Field(
            default=True,
            description="Enable or disable Python code execution."
        )

    def __init__(self):
        """
        Initializes the Pipeline with default or provided settings.
        """
        self.type = "filter"
        self.name = "Python Code Execution Filter"
        self.valves = self.Valves()

        # Cache for executed code results
        self.execution_cache = {}

    async def on_startup(self):
        """
        Called when the server starts.
        """
        print("Python Code Execution Filter started.")

    async def on_shutdown(self):
        """
        Called when the server shuts down.
        """
        print("Python Code Execution Filter stopped.")

    async def on_valves_updated(self):
        """
        Called when the valves configuration is updated.
        """
        print("Valves updated.")

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
        # Check cache first
        if code in self.execution_cache:
            return self.execution_cache[code]

        try:
            result = subprocess.run(
                ["python", "-c", code], capture_output=True, text=True, check=True
            )
            output = result.stdout.strip()
            self.execution_cache[code] = output
            return output
        except subprocess.CalledProcessError as e:
            error_message = f"Error executing code: {e.output.strip()}"
            self.execution_cache[code] = error_message
            return error_message

    async def inlet(self, body: Dict, user: Optional[Dict] = None, **kwargs) -> Dict:
        """
        Processes all incoming user messages.

        Parameters:
        ----------
        body : dict
            The input body containing user messages.
        user : Optional[dict]
            User-related metadata.

        Returns:
        -------
        dict
            Optionally modified input body.
        """
        print("Processing inlet...")

        messages = body.get("messages", [])
        if not messages:
            return body

        user_message = get_last_user_message(messages)
        print(f"User Message: {user_message}")

        # You can optionally modify the user message here
        # For example: Add context, sanitize input, etc.

        return body

    async def outlet(self, body: Dict, user: Optional[Dict] = None, **kwargs) -> Dict:
        """
        Inspects all outgoing assistant messages for Python code, executes it, and appends the results.

        Parameters:
        ----------
        body : dict
            The outgoing response body containing model-generated messages.
        user : Optional[dict]
            User-related metadata.

        Returns:
        -------
        dict
            Modified output body with code execution results, if applicable.
        """
        print("Processing outlet...")

        if not self.valves.ENABLE_EXECUTION:
            print("Code execution is disabled.")
            return body

        messages = body.get("messages", [])
        if not messages:
            return body

        assistant_message = get_last_assistant_message(messages)
        print(f"Assistant Message: {assistant_message}")

        # Check for Python code in the assistant message
        python_code_match = re.search(r"```python\n(.*?)```", assistant_message, re.DOTALL)

        if python_code_match:
            python_code = python_code_match.group(1)
            print(f"Detected Python code:\n{python_code}")

            # Execute the Python code
            execution_result = self.execute_python_code(python_code)
            print(f"Execution Result:\n{execution_result}")

            # Append the execution result to the assistant's response
            messages.append({
                "role": "assistant",
                "content": f"Executed Python Code:\n```\n{execution_result}\n```"
            })

        return {**body, "messages": messages}

# Example Usage
if __name__ == "__main__":
    import asyncio

    pipeline_instance = Pipeline()

    # Simulated model response with Python code
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

    # Process the input through inlet (pre-processing)
    processed_input = asyncio.run(pipeline_instance.inlet(model_output))
    print("Processed Input:", processed_input)

    # Process the output through outlet (post-processing)
    processed_output = asyncio.run(pipeline_instance.outlet(processed_input))
    print("Processed Output:", processed_output)
