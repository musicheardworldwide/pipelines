"""
title: Filter Template
author: Wes Caldwell
version: 1.0
license: MIT
description: A filter pipeline template for Open WebUI.
requirements: 
environment_variables: 
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict

class Pipeline:
    """
    Generalized Filter Class Template for Open WebUI.

    Purpose:
    --------
    This template serves as a foundation for creating filter pipelines in Open WebUI.
    Filters are used to process and manipulate user input (inlet) and model output (outlet)
    to enhance data quality, provide context, or sanitize information.

    Components:
    -----------
    - Valves: Configurable settings for the filter.
    - inlet: Pre-processes user inputs before sending them to the model.
    - outlet: Post-processes model outputs before presenting them to the user.
    """

    class Valves(BaseModel):
        """
        Configuration options for the Filter.

        Attributes:
        -----------
        pipelines: List[str]
            Target pipeline IDs this filter connects to. Use ["*"] to connect to all pipelines.
        priority: int
            Priority of the filter. Lower numbers are higher priority.
        ENABLE_FILTER: bool
            Whether the filter is enabled.
        CUSTOM_MESSAGE: str
            A custom system message to add during inlet or outlet processing.
        """
        pipelines: List[str] = Field(default=["*"], description="Pipelines to connect to.")
        priority: int = Field(default=0, description="Filter priority.")
        ENABLE_FILTER: bool = Field(default=True, description="Enable or disable the filter.")
        CUSTOM_MESSAGE: str = Field(default="", description="Custom message to include in processing.")

    def __init__(self):
        """
        Initializes the Filter with default or configured valve settings.
        """
        self.type = "filter"
        self.name = "Generic Filter"
        self.valves = self.Valves()

    async def on_startup(self):
        """
        Called when the server starts up.
        """
        print(f"on_startup:{self.name}")

    async def on_shutdown(self):
        """
        Called when the server shuts down.
        """
        print(f"on_shutdown:{self.name}")

    async def on_valves_updated(self):
        """
        Called when the valves configuration is updated.
        """
        print(f"on_valves_updated:{self.name}")

    def inlet(self, body: Dict, **kwargs: Optional[Dict]) -> Dict:
        """
        Processes user input before sending it to the model.

        Parameters:
        ----------
        body : dict
            The incoming request body containing user messages.
        kwargs : dict
            Optional additional parameters, such as user information.

        Returns:
        -------
        dict
            The modified input body.
        """
        if not self.valves.ENABLE_FILTER:
            return body

        # Example: Add a custom system message for context
        if self.valves.CUSTOM_MESSAGE:
            context_message = {
                "role": "system",
                "content": self.valves.CUSTOM_MESSAGE,
            }
            body.setdefault("messages", []).insert(0, context_message)

        return body

    def outlet(self, body: Dict, **kwargs: Optional[Dict]) -> Dict:
        """
        Processes model output before presenting it to the user.

        Parameters:
        ----------
        body : dict
            The outgoing response body containing model-generated messages.
        kwargs : dict
            Optional additional parameters for further customization.

        Returns:
        -------
        dict
            The modified output body.
        """
        if not self.valves.ENABLE_FILTER:
            return body

        # Example: Append a custom footer to the model's response
        if self.valves.CUSTOM_MESSAGE:
            response_message = body.get("messages", [])[-1]["content"]
            body["messages"][-1]["content"] = f"{response_message}\n\n{self.valves.CUSTOM_MESSAGE}"

        return body

# Example Usage
if __name__ == "__main__":
    # Initialize the filter
    filter_instance = Pipeline()
    filter_instance.valves.ENABLE_FILTER = True
    filter_instance.valves.CUSTOM_MESSAGE = "Remember, you are a helpful assistant."

    # Simulate input processing
    user_input = {"messages": [{"role": "user", "content": "How do I bake a cake?"}]}
    processed_input = filter_instance.inlet(user_input)

    # Simulate output processing
    model_output = {"messages": [{"role": "assistant", "content": "Preheat your oven to 350°F."}]}
    processed_output = filter_instance.outlet(model_output)

    print("Processed Input:", processed_input)
    print("Processed Output:", processed_output)
