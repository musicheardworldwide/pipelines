"""
title: Conversation Logger Pipeline
author: Your Name
author_url: https://example.com/your-profile  # Recommended by Open WebUI standards
description: A pipeline for logging all conversations to a file for cleaning and model training.
version: 1.0.0  # Semantic versioning preferred
license: MIT
requirements: pydantic>=2.0, aiohttp>=3.9.0  # Specify minimum versions
compatibility: openwebui>=0.1.7  # Add Open WebUI version compatibility
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from utils.pipelines.main import get_last_user_message  # Optional, if needed for message parsing


class Pipeline:
    class Valves(BaseModel):
        """
        Configuration model for the pipeline.
        
        Attributes:
            pipelines: List of pipelines to connect to ('*' for all)
            priority: Execution priority (lower runs earlier)
            log_file_path: Path to store conversation logs
            model_to_override: Specific model to filter logging (empty for all)
            max_log_size: Maximum log file size in MB before rotation
        """
        pipelines: List[str] = ["*"]
        priority: int = 0
        log_file_path: str = "./conversation_logs.json"
        model_to_override: str = ""
        max_log_size: int = 10  # New parameter for log rotation

    def __init__(self):
        """Initialize the pipeline with default values."""
        self.type = "filter"  # Must be one of: 'filter', 'pipe', or 'action'
        self.name = "conversation_logger"  # Lowercase, underscore-separated
        self.valves = self.Valves()  # Configuration instance
        self.current_log_size = 0

    async def on_startup(self):
        """Initialize resources when the pipeline starts."""
        print(f"[{self.name}] Starting conversation logger pipeline")
        os.makedirs(os.path.dirname(self.valves.log_file_path), exist_ok=True)
        if not os.path.exists(self.valves.log_file_path):
            with open(self.valves.log_file_path, "w", encoding="utf-8") as f:
                json.dump([], f)  # Initialize with empty array for valid JSON

    async def on_shutdown(self):
        """Clean up resources when the pipeline stops."""
        print(f"[{self.name}] Shutting down conversation logger pipeline")

    async def _rotate_logs_if_needed(self):
        """Rotate logs if they exceed maximum size."""
        if os.path.exists(self.valves.log_file_path):
            size_mb = os.path.getsize(self.valves.log_file_path) / (1024 * 1024)
            if size_mb > self.valves.max_log_size:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                rotated_path = f"{self.valves.log_file_path}.{timestamp}"
                os.rename(self.valves.log_file_path, rotated_path)
                with open(self.valves.log_file_path, "w", encoding="utf-8") as f:
                    json.dump([], f)

    async def log_conversation(self, body: Dict[str, Any], user: Optional[Dict[str, Any]] = None) -> None:
        """
        Log conversation to file in structured JSON format.
        
        Args:
            body: The conversation body containing messages and model info
            user: Optional user information dictionary
        """
        await self._rotate_logs_if_needed()
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user": {
                "id": user.get("id", "unknown") if user else "unknown",
                "name": user.get("name", "anonymous") if user else "anonymous"
            },
            "model": body.get("model", "unknown"),
            "messages": body.get("messages", []),
            "metadata": {  # Additional useful information
                "message_count": len(body.get("messages", [])),
                "last_user_message": get_last_user_message(body) if hasattr(get_last_user_message, '__call__') else None
            }
        }

        try:
            # Read existing logs
            existing_logs = []
            if os.path.exists(self.valves.log_file_path):
                with open(self.valves.log_file_path, "r", encoding="utf-8") as f:
                    existing_logs = json.load(f)
            
            # Append new log entry
            existing_logs.append(log_entry)
            
            # Write back to file
            with open(self.valves.log_file_path, "w", encoding="utf-8") as f:
                json.dump(existing_logs, f, indent=2)
                
        except Exception as e:
            print(f"[{self.name}] Error logging conversation: {str(e)}")

    async def inlet(self, body: Dict[str, Any], user: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process incoming conversation data.
        
        Args:
            body: The conversation data
            user: Optional user information
            
        Returns:
            The unmodified conversation data (pass-through filter)
        """
        try:
            if isinstance(body, str):
                body = json.loads(body)
            
            # Check model filter if specified
            if not self.valves.model_to_override or body.get("model") == self.valves.model_to_override:
                await self.log_conversation(body, user)
                
        except Exception as e:
            print(f"[{self.name}] Error in inlet processing: {str(e)}")
            
        return body  # Always return the original body for pass-through
