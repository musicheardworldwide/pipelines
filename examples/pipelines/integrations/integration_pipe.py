"""
Universal LLM Integration Pipeline for Open WebUI.
Supports OpenAI-compatible APIs, safe local commands, DB query hooks, and streaming.
"""

import os
import json
import asyncio
from typing import List, Dict, Optional, AsyncGenerator, Union
from pydantic import BaseModel, Field
import aiohttp
from blueprints.function_calling_blueprint import Pipeline as FunctionCallingBlueprint


class Pipeline(FunctionCallingBlueprint):
    """
    LLM + External Integration Pipeline with DB query hook and command processing.
    """

    # --- Configuration: Runtime Parameters (Valves) ---
    class Valves(FunctionCallingBlueprint.Valves):
        LLM_API_KEY: str = Field(default="", description="API key for the target LLM")
        LLM_BASE_URL: str = Field(default="https://api.openai.com/v1", description="Base URL for chat/completions")
        ALLOWED_SCRIPTS: List[str] = Field(default=["volume"], description="Safe local scripts (e.g., AppleScript)")
        TIMEOUT: int = Field(default=30, description="HTTP timeout for LLM requests")
        stream: bool = Field(default=True, description="Enable streaming by default")
        DB_URL: str = Field(default="", description="Database connection string")

    def __init__(self):
        super().__init__()
        self.name = "Universal LLM Integration Pipeline"
        self.type = "pipe"
        self.valves = self.Valves(**{
            **self.valves.model_dump(),
            "LLM_API_KEY": os.getenv("LLM_API_KEY", ""),
        })
        self.session: Optional[aiohttp.ClientSession] = None

    # --- Lifecycle Management ---
    async def on_startup(self):
        self.session = aiohttp.ClientSession()

    async def on_shutdown(self):
        if self.session:
            await self.session.close()

    # --- Main Pipeline Logic ---
    async def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[Dict[str, Union[str, Dict]]],
        body: Dict[str, Any],
    ) -> AsyncGenerator[str, None]:
        if await self._handle_local_command(user_message):
            return

        async for chunk in self._call_llm(user_message, model_id, stream=body.get("stream", self.valves.stream)):
            yield chunk

    # --- Local Script + DB Command Handling ---
    async def _handle_local_command(self, command: str) -> bool:
        parts = command.strip().split(" ")

        # Run DB Query
        if command.startswith("query "):
            await self._run_db_query(command[6:].strip())
            return True

        # Volume Control
        if parts[0] in self.valves.ALLOWED_SCRIPTS:
            if parts[0] == "volume" and len(parts) > 1:
                await self._set_volume(int(parts[1]))
                return True

        return False

    async def _set_volume(self, level: int):
        """macOS AppleScript volume control."""
        proc = await asyncio.create_subprocess_shell(
            f"osascript -e 'set volume output volume {level}'"
        )
        await proc.wait()

    async def _run_db_query(self, query: str):
        """Stub for future database interaction."""
        print(f"Executing DB query: {query}")
        # Connect and query with async driver (e.g., asyncpg for PostgreSQL)
        # This is a placeholder until you implement DB logic

    # --- LLM API Integration ---
    async def _call_llm(
        self,
        prompt: str,
        model_id: str,
        stream: bool,
    ) -> AsyncGenerator[str, None]:
        url = f"{self.valves.LLM_BASE_URL}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.valves.LLM_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
        }

        try:
            async with self.session.post(
                url, headers=headers, json=payload, timeout=self.valves.TIMEOUT
            ) as response:
                if stream:
                    async for line in response.content:
                        decoded = line.decode().strip()
                        if decoded:
                            yield decoded
                else:
                    data = await response.json()
                    yield json.dumps(data)
        except Exception as e:
            yield json.dumps({"error": f"LLM request failed: {str(e)}"})
