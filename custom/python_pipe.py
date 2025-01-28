from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
import subprocess
import ast
import logging

class Pipeline:
    def __init__(self):
        self.name = "Python Code Pipeline"
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def on_startup(self):
        self.logger.info(f"on_startup:{__name__}")

    async def on_shutdown(self):
        self.logger.info(f"on_shutdown:{__name__}")

    def is_valid_python_code(self, code):
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def install_python_libraries(self, libraries: List[str]):
        try:
            subprocess.run(
                ["pip", "install"] + libraries, capture_output=True, text=True, check=True
            )
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to install libraries: {e}")
            return False

    def execute_python_code(self, code, interpreter="python", env_vars=None, input_data=None, input_file=None, output_file=None):
        try:
            if input_file:
                with open(input_file, 'r') as f:
                    input_data = f.read()

            process = subprocess.Popen(
                [interpreter, "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env_vars
            )
            stdout, stderr = process.communicate(input=input_data, timeout=10)

            if output_file:
                with open(output_file, 'w') as f:
                    f.write(stdout)

            return stdout.strip(), process.returncode
        except subprocess.TimeoutExpired:
            process.kill()
            return "Error: Code execution timed out", -1
        except subprocess.CalledProcessError as e:
            return f"Error: {e.output.strip()}", e.returncode
        except Exception as e:
            return f"Error: {str(e)}", -1

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        self.logger.info(f"pipe:{__name__}")
        self.logger.info(f"Messages: {messages}")
        self.logger.info(f"User Message: {user_message}")

        if body.get("title", False):
            self.logger.info("Title Generation")
            return "Python Code Pipeline"
        else:
            # Check if the user wants to install libraries
            if body.get("install_libraries", False):
                libraries = body.get("libraries", [])
                if libraries:
                    if not self.install_python_libraries(libraries):
                        return "Error: Failed to install libraries"

            # Split the user_message into multiple code blocks
            code_blocks = user_message.split("\n\n")
            results = []

            for code in code_blocks:
                if self.is_valid_python_code(code):
                    interpreter = body.get("interpreter", "python")
                    env_vars = body.get("env_vars", None)
                    input_data = body.get("input_data", None)
                    input_file = body.get("input_file", None)
                    output_file = body.get("output_file", None)

                    stdout, return_code = self.execute_python_code(
                        code, interpreter, env_vars, input_data, input_file, output_file
                    )
                    results.append(stdout)
                else:
                    results.append("Error: Invalid Python code")

            return "\n".join(results)
