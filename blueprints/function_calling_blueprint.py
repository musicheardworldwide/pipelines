"""
title: Function Calling Pipeline
author: AI Assistant
version: 1.0
license: MIT
description: A pipeline for function calling from an OpenWebUI instance.
"""

from typing import List, Optional
from pydantic import BaseModel
import os
import requests
import json
import logging

from utils.pipelines.main import (
    get_last_user_message,
    add_or_update_system_message,
    get_tools_specs,
)

# System prompt for function calling
DEFAULT_SYSTEM_PROMPT = (
    """Tools: {}

If a function tool doesn't match the query, return an empty string. Else, pick a
function tool, fill in the parameters from the function tool's schema, and
return it in the format {{ "name": \"functionName\", "parameters": {{ "key":
"value" }} }}. Only pick a function if the user asks.  Only return the object. Do not return any other text."
"""
)

logging.basicConfig(level=logging.INFO)


class Pipeline:
    class Valves(BaseModel):
        # List target pipeline ids (models) that this filter will be connected to.
        # If you want to connect this filter to all pipelines, you can set pipelines to ["*"]
        pipelines: List[str] = []

        # Assign a priority level to the filter pipeline.
        # The priority level determines the order in which the filter pipelines are executed.
        # The lower the number, the higher the priority.
        priority: int = 0

        # Valves for function calling
        OPENAI_API_BASE_URL: str
        OPENAI_API_KEY: str
        TASK_MODEL: str
        TEMPLATE: str

    def __init__(self, prompt: str | None = None) -> None:
        # Pipeline filters are only compatible with Open WebUI
        # You can think of filter pipeline as a middleware that can be used to edit the form data before it is sent to the OpenAI API.
        self.type = "filter"

        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "function_calling_blueprint"
        self.name = "Function Calling Blueprint"
        self.prompt = prompt or DEFAULT_SYSTEM_PROMPT
        self.tools: object = None

        # Initialize valves
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],  # Connect to all pipelines
                "OPENAI_API_BASE_URL": os.getenv(
                    "OPENAI_API_BASE_URL", "https://api.deepseek.com/v1"
                ),
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"),
                "TASK_MODEL": os.getenv("TASK_MODEL", "deepseek-chat"),
                "TEMPLATE": """Use the following context as your learned knowledge, inside <context></context> XML tags.
<context>
    {{CONTEXT}}
</context>

When answer to user:
- If you don't know, just say that you don't know.
- If you don't know when you are not sure, ask for clarification.
Avoid mentioning that you obtained the information from the context.
And answer according to the language of the user's question.""",
            }
        )

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

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """
        Process the incoming request and perform function calling.

        :param body: The request body.
        :param user: The user information.
        :return: The updated request body.
        """
        # If title generation is requested, skip the function calling filter
        if body.get("title", False):
            return body

        logging.info(f"pipe:{__name__}")
        logging.debug(f"User: {user}")

        # Get the last user message
        user_message = get_last_user_message(body["messages"])

        # Get the tools specs
        tools_specs = get_tools_specs(self.tools)

        prompt = self.prompt.format(json.dumps(tools_specs, indent=2))
        content = "History:\n" + "\n".join(
            [
                f"{message['role']}: {message['content']}"
                for message in body["messages"][::-1][:4]
            ]
        ) + f"Query: {user_message}"

        result = self.run_completion(prompt, content)
        messages = self.call_function(result, body["messages"])

        return {**body, "messages": messages}

    def call_function(self, result, messages: list[dict]) -> list[dict]:
        """
        Call the function specified in the result and update the messages.

        :param result: The result from the function calling.
        :param messages: The list of messages.
        :return: The updated list of messages.
        """
        if "name" not in result:
            return messages

        function = getattr(self.tools, result["name"])
        function_result = None
        try:
            function_result = function(**result["parameters"])
        except Exception as e:
            logging.error(f"Error calling function: {e}")

        # Add the function result to the system prompt
        if function_result:
            system_prompt = self.valves.TEMPLATE.replace(
                "{{CONTEXT}}", function_result
            )

            messages = add_or_update_system_message(system_prompt, messages)

            # Return the updated messages
            return messages

    def run_completion(self, system_prompt: str, content: str) -> dict:
        """
        Call the OpenAI API to get the function response.

        :param system_prompt: The system prompt.
        :param content: The user content.
        :return: The function response.
        """
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
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": content,
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
                logging.debug(f"Function response: {result}")
                return result

        except Exception as e:
            logging.error(f"Error: {e}")

            if r:
                try:
                    logging.error(f"API response: {r.json()}")
                except:
                    pass

        return {}


# Usage Example
if __name__ == "__main__":
    pipeline = Pipeline()
    body = {
        "messages": [
            {"role": "user", "content": "What is the weather like in New York?"}
        ],
        "title": False,
    }
    user = {"id": "123"}
    result = pipeline.inlet(body, user)
    print(result)
