"""
title: Command Execution Filter Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for detecting and executing Python, Bash, or cURL commands in chat messages.
requirements: None
"""

from typing import List, Optional
from schemas import OpenAIChatMessage
from pydantic import BaseModel
import subprocess
import shlex
import re


class Pipeline:
    class Valves(BaseModel):
        # List target pipeline ids (models) that this filter will be connected to.
        # Use ["*"] to connect to all pipelines.
        pipelines: List[str] = []

        # Assign a priority level to the filter pipeline.
        # The lower the number, the higher the priority.
        priority: int = 0

    def __init__(self):
        # Pipeline filters are only compatible with Open WebUI
        self.type = "filter"

        # Optionally, you can set the id and name of the pipeline.
        self.name = "Command Execution Filter"

        # Initialize valves
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],  # Connect to all pipelines
            }
        )

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    async def on_valves_updated(self):
        # This function is called when the valves are updated.
        pass

    def execute_python_code(self, code: str) -> str:
        """
        Execute Python code using subprocess.
        """
        try:
            result = subprocess.run(
                ["python", "-c", code], capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            return f"Error: {e.output.strip()}"

    def execute_bash_command(self, command: str) -> str:
        """
        Execute a Bash command using subprocess.
        """
        try:
            result = subprocess.run(
                shlex.split(command), capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            return f"Error: {e.output.strip()}"

    def execute_curl_command(self, command: str) -> str:
        """
        Execute a cURL command using subprocess.
        """
        try:
            result = subprocess.run(
                shlex.split(command), capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            return f"Error: {e.output.strip()}"

    def detect_and_execute(self, message: str) -> Optional[str]:
        """
        Detect if the message contains Python, Bash, or cURL commands and execute them.
        """
        # Detect Python code
        python_pattern = r"```python\s*(.*?)\s*```"
        python_match = re.search(python_pattern, message, re.DOTALL)
        if python_match:
            code = python_match.group(1).strip()
            return self.execute_python_code(code)

        # Detect Bash command
        bash_pattern = r"```bash\s*(.*?)\s*```"
        bash_match = re.search(bash_pattern, message, re.DOTALL)
        if bash_match:
            command = bash_match.group(1).strip()
            return self.execute_bash_command(command)

        # Detect cURL command
        curl_pattern = r"```curl\s*(.*?)\s*```"
        curl_match = re.search(curl_pattern, message, re.DOTALL)
        if curl_match:
            command = curl_match.group(1).strip()
            return self.execute_curl_command(command)

        # If no command is detected, return None
        return None

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """
        This filter is applied to the form data before it is sent to the OpenAI API.
        """
        print(f"inlet:{__name__}")

        # Extract the user's message
        user_message = body["messages"][-1]["content"]

        # Detect and execute commands
        result = self.detect_and_execute(user_message)
        if result:
            # Replace the user's message with the command result
            body["messages"][-1]["content"] = result
            print(f"Command executed. Result: {result}")

        return body
