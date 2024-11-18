"""
title: Python Code Execution Pipeline
author: AI Assistant
version: 1.0
license: MIT
description: A pipeline for executing Python code from an OpenWebUI instance.

"""

import subprocess
import logging
from typing import List, Union, Generator, Iterator

logging.basicConfig(level=logging.INFO)


class Pipeline:
    def __init__(self):
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "python_code_pipeline"
        self.name = "Python Code Pipeline"
        pass

    async def on_startup(self):
        """
        This function is called when the server is started.
        """
        logging.info(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        """
        This function is called when the server is stopped.
        """
        logging.info(f"on_shutdown:{__name__}")
        pass

    def execute_python_code(self, code: str, python_version: str = "python3") -> tuple:
        """
        Execute Python code using the specified Python version.

        :param code: The Python code to execute.
        :param python_version: The Python version to use (default is "python3").
        :return: A tuple containing the stdout and return code.
        """
        try:
            result = subprocess.run(
                [python_version, "-c", code],
                capture_output=True,
                text=True,
                check=True,
            )
            stdout = result.stdout.strip()
            return stdout, result.returncode
        except subprocess.CalledProcessError as e:
            logging.error(f"Error executing Python code: {e}")
            return e.output.strip(), e.returncode

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        """
        Execute Python code from the user message.

        :param user_message: The user's message containing Python code.
        :param model_id: The model ID.
        :param messages: The list of messages.
        :param body: The request body.
        :return: The output of the executed Python code.
        """
        logging.info(f"pipe:{__name__}")

        logging.debug(f"Messages: {messages}")
        logging.debug(f"User Message: {user_message}")

        if body.get("title", False):
            logging.info("Title Generation")
            return "Python Code Pipeline"
        else:
            stdout, return_code = self.execute_python_code(user_message)
            if return_code != 0:
                logging.error(f"Python code execution failed with return code {return_code}")
            return stdout


# Usage Example
if __name__ == "__main__":
    pipeline = Pipeline()
    user_message = "print('Hello, World!')"
    result = pipeline.pipe(user_message, "model_id", [], {"title": False})
    print(result)
