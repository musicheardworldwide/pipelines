"""
Universal Filter Pipeline for Open WebUI (v0.6+)
"""

import os
from datetime import datetime
from typing import Dict, Optional, Any
import requests
from pydantic import BaseModel, Field
from blueprints.function_calling_blueprint import Pipeline as FunctionCallingBlueprint

class Pipeline(FunctionCallingBlueprint):
    """
    A functional, extensible filter pipeline template aligned with Open WebUI standards.
    """

    # --- Runtime Configuration (Valves) ---
    class Valves(FunctionCallingBlueprint.Valves):
        SERVICE_API_KEY: str = Field(default="", description="API key for external service")
        SERVICE_BASE_URL: str = Field(default="", description="Base URL for external API")
        MODEL_OVERRIDE: str = Field(default="", description="Target model for interception")
        PRIORITY: int = Field(default=0, description="Filter pipeline priority")
        pipelines: list = Field(default=["*"], description="Targeted pipelines")

    # --- Toolset Logic (Stateless Methods) ---
    class Tools:
        def __init__(self, pipeline: "Pipeline"):
            self.pipeline = pipeline

        def fetch_data(self, query: str) -> Dict[str, Any]:
            key = self.pipeline.valves.SERVICE_API_KEY
            base = self.pipeline.valves.SERVICE_BASE_URL
            if not key or not base:
                return {"error": "Missing service configuration."}
            try:
                response = requests.get(f"{base}/endpoint", params={"q": query, "api_key": key})
                response.raise_for_status()
                return response.json()
            except Exception as e:
                return {"error": str(e)}

        def process_input(self, text: str) -> str:
            return f"[Filtered]: {text.strip()}"

        def get_time(self) -> str:
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def __init__(self):
        super().__init__()
        self.name = "Universal Filter Pipeline"
        self.type = "filter"
        self.valves = self.Valves(**{
            **self.valves.model_dump(),
            "SERVICE_API_KEY": os.getenv("SERVICE_API_KEY", ""),
        })
        self.tools = self.Tools(self)

    # --- Optional Inlet Filter ---
    async def inlet(self, body: Dict[str, Any], user: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self.valves.MODEL_OVERRIDE and body.get("model") == self.valves.MODEL_OVERRIDE:
            body["input"] = self.tools.process_input(body.get("input", ""))
        return body

    # --- Optional Stream Filter ---
    async def stream(self, token: str, body: Dict[str, Any], user: Optional[Dict[str, Any]] = None) -> str:
        return token  # Optionally modify streamed output tokens

    # --- Optional Outlet Filter ---
    async def outlet(self, body: Dict[str, Any], user: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return body  # Optionally post-process entire response
