"""
title: Conversation Logger
author: Your Name
description: Logs all conversations to JSON files for analysis and training
version: 1.0.0
license: MIT
requirements: pydantic
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel

class Valves(BaseModel):
    """Control settings for the conversation logger"""
    log_dir: str = "./logs/conversations"
    max_file_size: int = 10  # MB
    file_prefix: str = "conversation_"
    retain_files: int = 30  # Number of log files to keep

class Filter:
    def __init__(self):
        self.type = "filter"
        self.name = "conversation_logger"
        self.valves = Valves()
        self.current_file = None
        self.current_size = 0

    async def on_startup(self):
        """Initialize logging directory"""
        os.makedirs(self.valves.log_dir, exist_ok=True)
        self._rotate_file()

    async def on_shutdown(self):
        """Clean up resources"""
        if self.current_file:
            self.current_file.close()

    def _get_current_filepath(self):
        """Generate timestamped filename"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(
            self.valves.log_dir,
            f"{self.valves.file_prefix}{timestamp}.jsonl"
        )

    def _rotate_file(self):
        """Rotate log file when size limit reached"""
        if self.current_file:
            self.current_file.close()

        # Clean up old files if over retention limit
        files = sorted([
            f for f in os.listdir(self.valves.log_dir) 
            if f.startswith(self.valves.file_prefix)
        ])
        while len(files) >= self.valves.retain_files:
            os.remove(os.path.join(self.valves.log_dir, files.pop(0)))

        self.current_filepath = self._get_current_filepath()
        self.current_file = open(self.current_filepath, "a", encoding="utf-8")
        self.current_size = 0

    async def inlet(self, body: Dict[str, Any], user: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Log incoming conversation data"""
        return await self._log_conversation(body, user, "inlet")

    async def outlet(self, body: Dict[str, Any], user: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Log outgoing conversation data""" 
        return await self._log_conversation(body, user, "outlet")

    async def _log_conversation(self, body: Dict[str, Any], user: Optional[Dict[str, Any]], direction: str) -> Dict[str, Any]:
        """Shared logging logic"""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "direction": direction,
                "user": {
                    "id": user.get("id", "anonymous") if user else "anonymous",
                    "name": user.get("name", "anonymous") if user else "anonymous"
                },
                "model": body.get("model", "unknown"),
                "messages": body.get("messages", [])
            }

            log_line = json.dumps(log_entry) + "\n"
            log_size = len(log_line.encode('utf-8'))

            if self.current_size + log_size > self.valves.max_file_size * 1024 * 1024:
                self._rotate_file()

            self.current_file.write(log_line)
            self.current_file.flush()
            self.current_size += log_size

        except Exception as e:
            print(f"[{self.name}] Error logging conversation: {str(e)}")

        return body
