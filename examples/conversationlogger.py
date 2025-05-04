"""
title: Conversation Logger Pipeline
author: Your Name
date: 2024-06-18
version: 1.0
license: MIT
description: A pipeline for logging all conversations to a file for cleaning and model training.
requirements: pydantic, aiohttp, json
"""
from typing import Dict, Optional, List
from pydantic import BaseModel
import json
import os
from datetime import datetime
from utils.pipelines.main import get_last_user_message  # Optional, if needed for message parsing

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = ["*"]  # Connect to all pipelines by default
        priority: int = 0  # Lower priority means it runs earlier
        log_file_path: str = "./conversation_logs.json"  # Path to the log file
        model_to_override: str = ""  # Optional: Only log for specific models

    def __init__(self):
        self.type = "filter"
        self.name = "Conversation Logger"
        self.valves = self.Valves()

    async def on_startup(self):
        """Initialize resources (e.g., ensure log file exists)."""
        print(f"on_startup:{__name__}")
        os.makedirs(os.path.dirname(self.valves.log_file_path), exist_ok=True)
        if not os.path.exists(self.valves.log_file_path):
            with open(self.valves.log_file_path, "w") as f:
                f.write("")  # Create empty file

    async def on_shutdown(self):
        """Clean up resources."""
        print(f"on_shutdown:{__name__}")
        pass

    async def log_conversation(self, body: Dict[str, Any], user: Optional[Dict[str, Any]] = None) -> None:
        """Logs the conversation to a file."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user.get("id", "unknown") if user else "unknown",
            "model": body.get("model", "unknown"),
            "messages": body.get("messages", []),
        }
        with open(self.valves.log_file_path, "a") as log_file:
            log_file.write(json.dumps(log_entry) + "\n")

    async def inlet(self, body: Dict[str, Any], user: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Intercepts and logs the conversation."""
        print(f"pipe:{__name__}")
        if isinstance(body, str):
            body = json.loads(body)
        
        # Optional: Only log if the model matches `model_to_override`
        if not self.valves.model_to_override or body.get("model") == self.valves.model_to_override:
            await self.log_conversation(body, user)
        
        return body  # Pass the body through unchanged
