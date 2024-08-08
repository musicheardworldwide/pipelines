import os
import importlib.util
from typing import List, Optional
from pydantic import BaseModel
import json
import requests

from blueprints.function_calling_blueprint import Pipeline as FunctionCallingBlueprint

def get_last_user_message(messages: List[dict]) -> str:
    """
    Get the last message from the user in the list of messages.

    :param messages: List of message dictionaries.
    :return: The content of the last user message.
    """
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""

class Pipeline(FunctionCallingBlueprint):
    class Valves(FunctionCallingBlueprint.Valves):
        # Add your custom parameters here
        TOOLS_DIR: str = "/home/ui_data/tools"
        FUNCTIONS_DIR: str = "/home/ui_data/functions"

    class Tools:
        def __init__(self, pipeline) -> None:
            self.pipeline = pipeline
            self.load_tools()

        def load_tools(self):
            tools_dir = self.pipeline.valves.TOOLS_DIR
            if os.path.exists(tools_dir):
                for tool_file in os.listdir(tools_dir):
                    if tool_file.endswith('.py'):
                        module_name = tool_file[:-3]
                        module_path = os.path.join(tools_dir, tool_file)
                        spec = importlib.util.spec_from_file_location(module_name, module_path)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        for attr in dir(module):
                            if not attr.startswith('_'):
                                setattr(self, attr, getattr(module, attr))

    def __init__(self):
        super().__init__()
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "my_tools_pipeline"
        self.name = "My Tools Pipeline"
        self.valves = self.Valves(
            **{
                **self.valves.model_dump(),
                "pipelines": ["*"],  # Connect to all pipelines
            },
        )
        self.tools = self.Tools(self)

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        # If title generation is requested, skip the function calling filter
        if body.get("title", False):
            return body

        print(f"pipe:{__name__}")
        print(user)

        # Get the last user message
        user_message = get_last_user_message(body["messages"])

        # Get the tools specs
        tools_specs = get_tools_specs(self.tools)

        # System prompt for function calling
        fc_system_prompt = (
            f"Tools: {json.dumps(tools_specs, indent=2)}"
            + """
If a function tool doesn't match the query, return an empty string. Else, pick a function tool, fill in the parameters from the function tool's schema, and return it in the format { "name": \"functionName\", "parameters": { "key": "value" } }. Only pick a function if the user asks.  Only return the object. Do not return any other text."
"""
        )

        r = None
        try:
            # Call the OpenAI API to get the function response
            r = requests.post(
                url=f"{self.valves.OPENAI_API_BASE_URL}/chat/completions",
                json={
                    "model": self.valves.TASK_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": fc_system_prompt,
                        },
                        {
                            "role": "user",
                            "content": "History:\n"
                            + "\n".join(
                                [
                                    f"{message['role']}: {message['content']}"
                                    for message in body["messages"][::-1][:4]
                                ]
                            )
                            + f"Query: {user_message}",
                        },
                    ],
                    # TODO: dynamically add response_format?
                    # "response_format": {"type": "json_object"},
                },
                headers={
                    "Authorization": f"Bearer {self.valves.OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                stream=False,
            )
            r.raise_for_status()

            response = r.json()
            content = response["choices"][0]["message"]["content"]

            # Parse the function response
            if content != "":
                result = json.loads(content)
                print(result)

                # Call the function
                if "name" in result:
                    function = getattr(self.tools, result["name"])
                    function_result = None
                    try:
                        function_result = function(**result["parameters"])
                    except Exception as e:
                        print(e)

                    # Add the function result to the system prompt
                    if function_result:
                        system_prompt = self.valves.TEMPLATE.replace(
                            "{{CONTEXT}}", function_result
                        )

                        print(system_prompt)
                        messages = add_or_update_system_message(
                            system_prompt, body["messages"]
                        )

                        # Return the updated messages
                        return {**body, "messages": messages}

        except Exception as e:
            print(f"Error: {e}")

            if r:
                try:
                    print(r.json())
                except:
                    pass

        return body

def get_tools_specs(tools):
    """
    Get the specifications of the tools.

    :param tools: The tools object.
    :return: A list of tool specifications.
    """
    tools_specs = []
    for tool_name in dir(tools):
        if not tool_name.startswith('_'):
            tool = getattr(tools, tool_name)
            if callable(tool):
                tools_specs.append({
                    "name": tool_name,
                    "description": tool.__doc__.strip() if tool.__doc__ else "No description provided.",
                    "parameters": tool.__annotations__
                })
    return tools_specs

def add_or_update_system_message(system_prompt, messages):
    """
    Add or update the system message in the list of messages.

    :param system_prompt: The system prompt to add or update.
    :param messages: The list of messages.
    :return: The updated list of messages.
    """
    system_message_exists = False
    for i, message in enumerate(messages):
        if message.get("role") == "system":
            messages[i] = {"role": "system", "content": system_prompt}
            system_message_exists = True
            break
    if not system_message_exists:
        messages.append({"role": "system", "content": system_prompt})
    return messages
