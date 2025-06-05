"""
title: SIN Task Execution Pipe
author: Dr. Wes Caldwell & iA-SIN
description: Core pipe emulating SIN's root controller for task decomposition and execution, potentially using sequential_thinking concepts.
version: 0.1.0
license: MIT
requirements: pydantic, aiohttp # for potential internal LLM calls
"""

import os
import json
from typing import List, Union, Generator, Iterator, Dict, Any, Optional, AsyncGenerator
import aiohttp # If making internal LLM calls

# Assuming OpenAIChatMessage might be used for internal structuring if needed.
# from schemas import OpenAIChatMessage 

class Pipeline:
    class Valves(BaseModel):
        # Default LLM for internal reasoning/decomposition if not specified in body
        default_llm_model_for_reasoning: str = Field(default="gpt-3.5-turbo") 
        max_sequential_steps: int = Field(default=5)
        # This would be your actual LLM API key for SIN's internal reasoning, 
        # separate from the one used by OpenWebUI for the final user-facing LLM.
        # Or, it could point to a local ollama-mcp.
        internal_llm_api_key: str = Field(default=os.getenv("SIN_INTERNAL_LLM_API_KEY", ""))
        internal_llm_base_url: str = Field(default=os.getenv("SIN_INTERNAL_LLM_BASE_URL", "http://localhost:11434/v1")) # Example: local Ollama

    def __init__(self):
        self.type = "pipe"
        self.name = "SIN Task Execution Pipe"
        self.valves = self.Valves()
        self.session: Optional[aiohttp.ClientSession] = None
        print(f"Initialized {self.name} with valves: {self.valves.model_dump_json(indent=2)}")

    async def on_startup(self):
        print(f"on_startup:{self.name}")
        self.session = aiohttp.ClientSession()

    async def on_shutdown(self):
        print(f"on_shutdown:{self.name}")
        if self.session:
            await self.session.close()

    async def on_valves_updated(self):
        print(f"on_valves_updated:{self.name} - Valves updated: {self.valves.model_dump_json(indent=2)}")

    async def _internal_llm_call(self, messages: List[Dict[str, str]], model: str) -> str:
        """Helper to make a call to an internal LLM for reasoning/decomposition."""
        if not self.session:
            # Fallback or error if session not initialized
            print(f"{self.name}: Error - HTTP session not initialized for internal LLM call.")
            return "Error: Internal reasoning module not available."

        payload = {
            "model": model,
            "messages": messages,
            "stream": False # Typically non-streamed for internal logic
        }
        headers = {
            "Content-Type": "application/json",
        }
        # Add Authorization header if internal_llm_api_key is set (for OpenAI-like APIs)
        if self.valves.internal_llm_api_key and "openai.com" in self.valves.internal_llm_base_url: # crude check
             headers["Authorization"] = f"Bearer {self.valves.internal_llm_api_key}"


        api_url = f"{self.valves.internal_llm_base_url.rstrip('/')}/chat/completions"
        
        print(f"{self.name}: Making internal LLM call to {api_url} with model {model}")

        try:
            async with self.session.post(api_url, headers=headers, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                # Extract content from the first choice's message
                return result.get("choices", [{}])[0].get("message", {}).get("content", "No content from internal LLM.")
        except Exception as e:
            print(f"{self.name}: Error during internal LLM call: {e}")
            return f"Error in internal reasoning: {e}"

    async def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> AsyncGenerator[str, None]:
        print(f"pipe:{self.name} - User message: {user_message}")
        print(f"pipe:{self.name} - Full request body: {json.dumps(body, indent=2)}")

        # 1. Initial Decomposition (Conceptual sequential_thinking)
        # For a real implementation, this would be more sophisticated.
        # Here, we'll use a simple heuristic or a direct LLM call for decomposition.
        
        yield "SIN Task Execution: Analyzing request...\n"
        
        decomposition_prompt_messages = [
            {"role": "system", "content": f"You are a task decomposition engine for the SIN AI. Break down the following user request into a series of simple, actionable steps (max {self.valves.max_sequential_steps}). If the task is simple, return 'DIRECT_EXECUTION: [original request]'. Otherwise, return steps as a numbered list. Each step should be concise."},
            {"role": "user", "content": user_message}
        ]
        
        decomposed_plan_str = await self._internal_llm_call(
            decomposition_prompt_messages, 
            self.valves.default_llm_model_for_reasoning
        )
        
        yield f"SIN Task Execution: Decomposition Plan:\n{decomposed_plan_str}\n"

        if decomposed_plan_str.startswith("DIRECT_EXECUTION:"):
            # Simple task, pass directly to a "final processing" LLM (could be model_id from body)
            # This part would typically be what OpenWebUI does *after* this pipe.
            # For demonstration, we'll simulate a final response.
            final_prompt_messages = [
                {"role": "system", "content": "You are SIN, executing a direct task."},
                {"role": "user", "content": user_message} # or decomposed_plan_str.split(":",1)[1].strip()
            ]
            # In a real scenario, you might not call another LLM here but rather format
            # the output for the *actual* LLM that OpenWebUI will call next.
            # This pipe's role is more about the "plan" than the final execution.
            # However, to show output, we'll simulate.
            final_response = await self._internal_llm_call(
                final_prompt_messages,
                model_id # Use the model_id passed into the pipe for the final step
            )
            yield f"SIN Task Execution: Final Result (simulated direct execution):\n{final_response}\n"
        else:
            # Complex task with steps
            steps = [s.strip() for s in decomposed_plan_str.split('\n') if s.strip() and s[0].isdigit()]
            yield f"SIN Task Execution: Executing {len(steps)} steps...\n"
            
            step_results = []
            for i, step_text in enumerate(steps):
                yield f"Step {i+1}/{len(steps)}: {step_text} - Processing...\n"
                # Simulate executing each step (e.g., calling a tool, another LLM, etc.)
                # For this example, each step is just a query to an LLM.
                step_processing_messages = [
                    {"role": "system", "content": f"You are SIN, executing sub-task: {step_text}"},
                    {"role": "user", "content": f"Based on the overall goal '{user_message}', what is the result or output for this specific step: '{step_text}'?"}
                ]
                step_result = await self._internal_llm_call(
                    step_processing_messages,
                    model_id # Use the model_id passed into the pipe
                )
                yield f"Step {i+1} Result: {step_result}\n"
                step_results.append(step_result)
            
            # 3. Final Synthesis
            yield "SIN Task Execution: Synthesizing final response from step results...\n"
            synthesis_prompt_messages = [
                {"role": "system", "content": "You are SIN. Based on the original request and the results of the executed sub-tasks, provide a comprehensive final answer to Dr. Caldwell."},
                {"role": "user", "content": f"Original Request: {user_message}\n\nSub-task Results:\n" + "\n".join([f"- Step {j+1}: {res}" for j, res in enumerate(step_results)]) + "\n\nSynthesized Answer:"}
            ]
            final_synthesized_response = await self._internal_llm_call(
                synthesis_prompt_messages,
                model_id # Use the model_id passed into the pipe
            )
            yield f"SIN Task Execution: Final Synthesized Response:\n{final_synthesized_response}\n"