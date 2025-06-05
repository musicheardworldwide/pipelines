"""
title: SIN Manager Emulation Pipe
author: Dr. Wes Caldwell & iA-SIN
description: A developed pipe for emulating the core functions of a SIN Manager, including LLM-based interpretation and optional tool API calls.
version: 0.2.0
license: MIT
requirements: pydantic, aiohttp
"""

import os
import json
import logging # Added for better logging
from typing import List, Union, Generator, Iterator, Dict, Any, Optional, AsyncGenerator
import aiohttp

from pydantic import BaseModel, Field

# Setup logger for this pipeline
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Set to DEBUG for very verbose output

# Basic console handler if no other logging is configured
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class Pipeline:
    class Valves(BaseModel):
        manager_name: str = Field(
            default=os.getenv("SIN_MANAGER_NAME", "UnnamedSINManager"),
            description="The name of the SIN Manager this pipeline instance is emulating (e.g., 'Content Manager', 'Financial Manager')."
        )
        
        # --- Primary Tool Configuration (Optional) ---
        primary_tool_api_endpoint: Optional[str] = Field(
            default=os.getenv("SIN_MANAGER_TOOL_ENDPOINT"),
            description="Optional: HTTP(S) endpoint for the manager's primary external tool/API."
        )
        primary_tool_api_key: Optional[str] = Field(
            default=os.getenv("SIN_MANAGER_TOOL_API_KEY"),
            description="Optional: API key for the primary tool, if required. Used as a Bearer token."
        )
        tool_api_timeout: int = Field(
            default=30,
            description="Timeout in seconds for calls to the primary tool API."
        )

        # --- Internal LLM Configuration (for interpretation & synthesis) ---
        manager_llm_model: str = Field(
            default=os.getenv("SIN_MANAGER_LLM_MODEL", "llama3"), # Changed default to llama3
            description="The LLM model to be used for this manager's internal reasoning, interpretation, and synthesis tasks."
        )
        manager_llm_base_url: str = Field(
            default=os.getenv("SIN_MANAGER_LLM_BASE_URL", "http://docker.host.internal:11434/v1"), # Default to Ollama via Docker host
            description="Base URL for the LLM API (e.g., Ollama, OpenAI-compatible)."
        )
        manager_llm_api_key: Optional[str] = Field(
            default=os.getenv("SIN_MANAGER_LLM_API_KEY"),
            description="Optional: API key for the manager's LLM, if it's a commercial API like OpenAI."
        )
        manager_llm_timeout: int = Field(
            default=60,
            description="Timeout in seconds for calls to the manager's LLM."
        )
        
        # --- Task Processing Configuration ---
        interpretation_prompt_template: str = Field(
            default="You are the {manager_name}, a specialized AI agent within the SIN (Symbiotic Intelligence Nexus) framework, serving Dr. Wes Caldwell. Your current task is to interpret the following user request: '{user_message}'.\n\n"
                    "Based on your function as {manager_name}, determine the necessary parameters or sub-tasks. If this request involves your primary tool ({primary_tool_description}), structure the parameters as a JSON object. For example: {{\"param1\": \"value1\", \"param2\": 123}}. \n"
                    "If no specific tool parameters are needed, or if the task is a direct query for you to answer, describe the core action or information to be processed. If the request is outside your scope, state that clearly.\n\n"
                    "Interpretation (JSON parameters or action description):"
        )
        primary_tool_description: str = Field(
            default="a generic API endpoint", # User should override this if a tool is configured
            description="A brief description of what the primary_tool_api_endpoint does, used in prompts."
        )

    def __init__(self):
        self.type = "pipe"
        self.valves = self.Valves() # Initialize with defaults or env vars
        self.name = f"SIN {self.valves.manager_name} Emulator" # Dynamic name
        
        self.session: Optional[aiohttp.ClientSession] = None
        logger.info(f"Initialized {self.name} with valves: {self.valves.model_dump_json(indent=2)}")

    async def on_startup(self):
        self.name = f"SIN {self.valves.manager_name} Emulator" # Ensure name is updated on startup
        logger.info(f"on_startup:{self.name}")
        self.session = aiohttp.ClientSession()
        # Initialize any specific "tools" or clients this manager needs based on valves

    async def on_shutdown(self):
        logger.info(f"on_shutdown:{self.name}")
        if self.session:
            await self.session.close()

    async def on_valves_updated(self):
        self.name = f"SIN {self.valves.manager_name} Emulator" # Update name if manager_name valve changes
        logger.info(f"on_valves_updated:{self.name} - Valves updated: {self.valves.model_dump_json(indent=2)}")
        # Potentially re-initialize clients if they depend on valve values (e.g., API keys, endpoints)
        # For aiohttp.ClientSession, it's generally fine unless base URLs or global headers change.

    async def _call_manager_tool_api(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Calls the configured primary tool API."""
        if not self.session:
            logger.error(f"{self.name}: HTTP session not available for tool API call.")
            return {"error": "HTTP session not available."}
        
        headers = {"Content-Type": "application/json"}
        if self.valves.primary_tool_api_key:
            headers["Authorization"] = f"Bearer {self.valves.primary_tool_api_key}"

        logger.info(f"{self.name}: Calling tool API at {endpoint} with payload: {json.dumps(payload)}")
        try:
            async with self.session.post(endpoint, headers=headers, json=payload, timeout=self.valves.tool_api_timeout) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientResponseError as e:
            logger.error(f"{self.name}: Tool API HTTP error {e.status} for {endpoint}: {e.message}", exc_info=True)
            try:
                error_content = await e.response.text()
                return {"error": f"Tool API HTTP error {e.status}: {e.message}", "details": error_content}
            except Exception:
                return {"error": f"Tool API HTTP error {e.status}: {e.message}"}
        except Exception as e:
            logger.error(f"{self.name}: Tool API call to {endpoint} failed: {e}", exc_info=True)
            return {"error": f"Tool API call failed: {str(e)}"}

    async def _manager_specific_llm_call(self, messages: List[Dict[str,str]], stream: bool = False) -> Union[str, AsyncGenerator[str, None]]:
        """Calls the manager's configured LLM."""
        if not self.session:
            logger.error(f"{self.name}: HTTP session not available for LLM call.")
            if stream: async def _s_err(): yield "Error: HTTP session not available."; return; return _s_err() # type: ignore
            return "Error: HTTP session not available."

        payload = {"model": self.valves.manager_llm_model, "messages": messages, "stream": stream}
        api_url = f"{self.valves.manager_llm_base_url.rstrip('/')}/chat/completions"
        
        headers = {"Content-Type": "application/json"}
        if self.valves.manager_llm_api_key: # Check if API key is provided
            headers["Authorization"] = f"Bearer {self.valves.manager_llm_api_key}"

        logger.info(f"{self.name}: Calling LLM API at {api_url} for model {self.valves.manager_llm_model} (stream={stream})")
        
        try:
            async with self.session.post(api_url, headers=headers, json=payload, timeout=self.valves.manager_llm_timeout) as response:
                response.raise_for_status()
                if stream:
                    async def _s_gen():
                        async for line in response.content:
                            if line:
                                line_str = line.decode('utf-8').strip()
                                if line_str.startswith("data: "):
                                    line_str = line_str[len("data: "):]
                                if line_str == "[DONE]":
                                    break
                                try:
                                    chunk_data = json.loads(line_str)
                                    content = chunk_data.get("choices", [{}])[0].get("delta", {}).get("content")
                                    if content:
                                        yield content
                                except json.JSONDecodeError:
                                    logger.warning(f"{self.name}: LLM stream - could not decode JSON from line: {line_str}")
                                except Exception as e_inner:
                                    logger.warning(f"{self.name}: LLM stream - error processing chunk: {e_inner} from line: {line_str}")
                    return _s_gen()
                else:
                    result = await response.json()
                    return result.get("choices", [{}])[0].get("message", {}).get("content", "No content from LLM.")
        except aiohttp.ClientResponseError as e:
            logger.error(f"{self.name}: LLM API HTTP error {e.status} for {api_url}: {e.message}", exc_info=True)
            error_detail = "Could not retrieve error body."
            try: error_detail = await e.response.text()
            except: pass
            err_msg = f"LLM API HTTP error {e.status}: {e.message}. Details: {error_detail}"
            if stream: async def _s_err_http(): yield err_msg; return; return _s_err_http() # type: ignore
            return err_msg
        except Exception as e:
            logger.error(f"{self.name}: LLM API call to {api_url} failed: {e}", exc_info=True)
            err_msg = f"LLM call failed: {str(e)}"
            if stream: async def _s_err_gen(): yield err_msg; return; return _s_err_gen() # type: ignore
            return err_msg


    async def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> AsyncGenerator[str, None]:
        logger.info(f"pipe:{self.name} - User message: '{user_message}', Model ID (ignored by this pipe): '{model_id}'")
        
        yield f"Executing with SIN Manager: {self.valves.manager_name} (Emulated)\n"
        yield f"Received Task: {user_message}\n\n"

        # 1. Interpretation Step
        yield "Phase 1: Interpreting request...\n"
        interpretation_prompt_str = self.valves.interpretation_prompt_template.format(
            manager_name=self.valves.manager_name,
            user_message=user_message,
            primary_tool_description=self.valves.primary_tool_description
        )
        interpretation_messages = [
            {"role": "system", "content": "You are an internal SIN component. Follow instructions precisely."},
            {"role": "user", "content": interpretation_prompt_str}
        ]
        
        interpretation_result_str = await self._manager_specific_llm_call(interpretation_messages, stream=False)
        if isinstance(interpretation_result_str, AsyncGenerator): # Should not happen with stream=False
            interpretation_result_str = "".join([chunk async for chunk in interpretation_result_str])
            
        yield f"Interpretation Result: {interpretation_result_str}\n\n"

        tool_params_dict: Optional[Dict[str, Any]] = None
        action_description: str = interpretation_result_str # Default to full interpretation if not JSON

        try:
            # Attempt to parse as JSON if it looks like JSON (heuristic)
            if interpretation_result_str.strip().startswith("{") and interpretation_result_str.strip().endswith("}"):
                tool_params_dict = json.loads(interpretation_result_str)
                action_description = f"Parsed tool parameters: {json.dumps(tool_params_dict)}"
                logger.info(f"{self.name}: Successfully parsed interpretation as JSON: {tool_params_dict}")
            else:
                logger.info(f"{self.name}: Interpretation result is not JSON, treating as action description.")
        except json.JSONDecodeError as e:
            logger.warning(f"{self.name}: Failed to parse interpretation result as JSON: {e}. Treating as raw text.")
            action_description = interpretation_result_str # Fallback to raw text

        # 2. Optional Tool Call Step
        tool_api_result_str = "No tool API call configured or required for this interpretation."
        if self.valves.primary_tool_api_endpoint and tool_params_dict:
            yield f"Phase 2: Calling Primary Tool API ({self.valves.primary_tool_description})...\n"
            tool_api_result = await self._call_manager_tool_api(self.valves.primary_tool_api_endpoint, tool_params_dict)
            tool_api_result_str = json.dumps(tool_api_result, indent=2)
            yield f"Tool API Result: {tool_api_result_str}\n\n"
        elif self.valves.primary_tool_api_endpoint and not tool_params_dict:
            yield "Phase 2: Primary Tool API configured, but interpretation did not yield valid JSON parameters. Skipping tool call.\n\n"
            tool_api_result_str = "Tool call skipped due to missing/invalid parameters from interpretation phase."


        # 3. Final Synthesis Step
        yield "Phase 3: Synthesizing final response...\n"
        synthesis_system_prompt = (
            f"You are {self.valves.manager_name}, a specialized AI agent within SIN. "
            f"Your original task from Dr. Wes Caldwell was: '{user_message}'.\n"
            f"Your interpretation of this task (or parameters for a tool) was: '{action_description}'.\n"
            f"The result from your primary tool (if called) was: '{tool_api_result_str}'.\n\n"
            "Based on all the above, provide a comprehensive final response, status update, or the direct answer to Dr. Caldwell. "
            "If an error occurred in a previous step, acknowledge it and explain the situation clearly."
        )
        synthesis_messages = [
            {"role": "system", "content": synthesis_system_prompt},
            {"role": "user", "content": "Please provide your final synthesized output for Dr. Caldwell."}
        ]
        
        # Allow final synthesis to stream if the LLM supports it
        final_response_gen = await self._manager_specific_llm_call(synthesis_messages, stream=True)
        if isinstance(final_response_gen, str): # Fallback if streaming failed or was not possible
            yield final_response_gen
        else:
            async for chunk in final_response_gen:
                yield chunk
        
        yield f"\n\n--- {self.valves.manager_name} Emulation Complete ---"
