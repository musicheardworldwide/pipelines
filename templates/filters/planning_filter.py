from typing import List, Union, Generator, Iterator, Optional
from pydantic import BaseModel, Field
import requests
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Filter:
    """
    Multi-model processing filter for Open WebUI.
    Processes user prompts using a pipeline of planning, verification, and execution models.
    """

    class Valves(BaseModel):
        """
        Configuration options for the MultiModelFilter.

        Attributes:
        -----------
        ENABLE_PIPELINE : bool
            Whether to enable the multi-model pipeline.
        MODELS : dict
            A dictionary containing the names, URLs, and API keys for the models in the pipeline.
        REQUEST_TIMEOUT : int
            Timeout for model requests in seconds.
        """
        ENABLE_PIPELINE: bool = Field(
            default=True,
            description="Enable or disable the multi-model pipeline."
        )
        MODELS: dict = Field(
            default={
                "planning": {
                    "url": os.environ.get("PLANNING_MODEL_URL", "http://planning-model-endpoint/api"),
                    "api_key": os.environ.get("PLANNING_MODEL_API_KEY", "")
                },
                "verification": {
                    "url": os.environ.get("VERIFICATION_MODEL_URL", "http://verification-model-endpoint/api"),
                    "api_key": os.environ.get("VERIFICATION_MODEL_API_KEY", "")
                },
                "main": {
                    "url": os.environ.get("MAIN_MODEL_URL", "http://main-model-endpoint/api"),
                    "api_key": os.environ.get("MAIN_MODEL_API_KEY", "")
                }
            },
            description="Dictionary containing model configurations: name, URL, and API key."
        )
        REQUEST_TIMEOUT: int = Field(
            default=30,
            description="Request timeout in seconds."
        )

    def __init__(self):
        """
        Initializes the MultiModelFilter with default or configured values.
        """
        self.valves = self.Valves()
        self.model_mappings = self.valves.MODELS

    def set_model_config(self, model_name: str, model_url: str, api_key: str) -> None:
        """
        Dynamically sets the URL and API key for a specific model.

        Parameters:
        ----------
        model_name : str
            Name of the model (e.g., "planning", "verification", "main").
        model_url : str
            URL for the model endpoint.
        api_key : str
            API key for the model.
        """
        if model_name in self.model_mappings:
            self.model_mappings[model_name]["url"] = model_url
            self.model_mappings[model_name]["api_key"] = api_key
            masked_key = '*' * (len(api_key) - 4) + api_key[-4:]
            logger.info(f"Updated {model_name} model configuration: URL={model_url}, API Key={masked_key}")
        else:
            logger.error(f"Invalid model name: {model_name}")

    def send_to_model(self, model_name: str, payload: dict) -> dict:
        """
        Sends a payload to a specified model and returns the response.

        Parameters:
        ----------
        model_name : str
            Name of the model (e.g., "planning", "verification", "main").
        payload : dict
            Payload to send to the model.

        Returns:
        -------
        dict
            Response from the model.
        """
        if model_name not in self.model_mappings:
            raise ValueError(f"Invalid model name: {model_name}")

        model_url = self.model_mappings[model_name]["url"]
        api_key = self.model_mappings[model_name]["api_key"]

        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        try:
            response = requests.post(model_url, json=payload, headers=headers, timeout=self.valves.REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Request to {model_name} model at {model_url} failed: {e}")
            return {"error": str(e)}

    def process_pipeline(self, user_message: str) -> str:
        """
        Processes the user's prompt through the planning, verification, and execution models.

        Parameters:
        ----------
        user_message : str
            The user's original prompt.

        Returns:
        -------
        str
            The final response after processing through all models.
        """
        if not self.valves.ENABLE_PIPELINE:
            return "Pipeline is disabled."

        # Step 1: Planning Model
        planning_payload = {
            "messages": [
                {"role": "system", "content": "You are a planning assistant."},
                {"role": "user", "content": user_message}
            ]
        }
        plan_response = self.send_to_model("planning", planning_payload)
        if "error" in plan_response:
            return f"Planning model error: {plan_response['error']}"

        generated_plan = plan_response.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not generated_plan:
            return "Planning model failed to generate a plan."

        # Step 2: Verification Model
        verification_payload = {
            "messages": [
                {"role": "system", "content": "You are a verification assistant."},
                {"role": "user", "content": f"Verify this plan: {generated_plan}"},
                {"role": "user", "content": f"Does this match the prompt: {user_message}"}
            ]
        }
        verification_response = self.send_to_model("verification", verification_payload)
        if "error" in verification_response:
            return f"Verification model error: {verification_response['error']}"

        verification_result = verification_response.get("choices", [{}])[0].get("message", {}).get("content", "")
        if "invalid" in verification_result.lower():
            return f"Plan verification failed: {verification_result}"

        # Step 3: Main Model
        execution_payload = {
            "messages": [
                {"role": "system", "content": "You are an execution assistant."},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": generated_plan}
            ]
        }
        execution_response = self.send_to_model("main", execution_payload)
        if "error" in execution_response:
            return f"Main model error: {execution_response['error']}"

        final_response = execution_response.get("choices", [{}])[0].get("message", {}).get("content", "")
        return final_response

    def inlet(self, body: dict, **kwargs) -> dict:
        """
        Processes user input (no changes in this implementation).

        Parameters:
        ----------
        body : dict
            Input body containing user messages.

        Returns:
        -------
        dict
            Unmodified input body.
        """
        return body

    def outlet(self, body: dict, **kwargs) -> dict:
        """
        Processes the output through the multi-model pipeline.

        Parameters:
        ----------
        body : dict
            Outgoing response body.

        Returns:
        -------
        dict
            Modified response body with pipeline results.
        """
        if not self.valves.ENABLE_PIPELINE:
            return body

        user_message = body.get("messages", [])[-1].get("content", "")
        pipeline_result = self.process_pipeline(user_message)

        # Add the pipeline result to the response
        body["messages"].append({
            "role": "assistant",
            "content": pipeline_result
        })
        return body

# Example Usage
if __name__ == "__main__":
    filter_instance = Filter()

    # Dynamically set a custom model configuration
    filter_instance.set_model_config("main", "https://example.com/api", "98w82")

    # Simulated user prompt
    input_body = {
        "messages": [
            {"role": "user", "content": "Update the server configuration."}
        ]
    }

    # Process the prompt through the pipeline
    output_body = filter_instance.outlet(input_body)
    print("Processed Output:", output_body)
