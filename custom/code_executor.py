import re
import subprocess
import shlex
from typing import List, Union
from schemas import OpenAIChatMessage


class filter:
    def __init__(self):
        self.name = "Command Execution Filter"
        print(f"Initialized filter: {self.name}")

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

    def detect_and_execute(self, message: str) -> Union[str, None]:
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

    async def on_message(self, message: OpenAIChatMessage) -> Union[str, None]:
        """
        Hook into the chat system to process messages in the background.
        """
        print(f"Filter processing message: {message['content']}")

        # Detect and execute commands
        result = self.detect_and_execute(message["content"])
        if result:
            print(f"Command executed. Result: {result}")
            return result

        # If no command is detected, return None to allow the message to proceed
        return None
