"""
title: Prompt Enhancer Filter
author: Open WebUI Pipeline Expert
description: Enhances user prompts by adding configurable system-level personas/instructions and prompt affixes, based on prompt engineering best practices.
version: 1.0.0
license: MIT
requirements: pydantic
"""

import os
import json # For pretty printing in logs
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = Field(
            default=["*"], 
            description="List of pipeline IDs to target with this filter. '*' applies to all."
        )
        priority: int = Field(
            default=0, 
            description="Execution priority for this filter. Lower numbers run earlier."
        )

        system_instruction: Optional[str] = Field(
            default="You are a helpful and insightful AI assistant. Your primary goal is to provide accurate, comprehensive, and clearly articulated responses. Strive to be engaging and adapt your communication style to be professional yet approachable.",
            description="A system-level instruction or persona definition that guides the LLM's overall behavior and tone. This is typically prepended or set as the initial system message."
        )
        user_prompt_prefix: Optional[str] = Field(
            default="",
            description="Optional text to automatically prepend to every user's core message content. Useful for adding consistent context or commands before the user's input."
        )
        user_prompt_suffix: Optional[str] = Field(
            default="",
            description="Optional text to automatically append to every user's core message content. Useful for adding consistent formatting requests or closing instructions after the user's input."
        )
        # Example for a future, more specific rule based on "Prompt Engineering.md":
        # output_format_guideline: Optional[str] = Field(
        #     default="",
        #     description="Specific guideline for the LLM's output format, e.g., 'Ensure all code examples are enclosed in triple backticks with the language specified.'"
        # )

    def __init__(self):
        self.type = "filter"
        self.name = "Prompt Enhancer Filter"
        
        # Initialize valves with default values.
        # These can be overridden by Open WebUI's configuration mechanism.
        self.valves = self.Valves()
        # Example of loading a default from an environment variable:
        # self.valves = self.Valves(
        #     system_instruction=os.getenv("PROMPT_ENHANCER_SYSTEM_INSTRUCTION", self.Valves().system_instruction),
        #     # ... other valves
        # )
        print(f"Initialized {self.name} with initial valves: {self.valves.model_dump_json(indent=2)}")

    async def on_startup(self):
        print(f"on_startup:{self.name}")
        # Placeholder for any setup needed when the pipeline starts, e.g., loading templates from files.

    async def on_shutdown(self):
        print(f"on_shutdown:{self.name}")
        # Placeholder for any cleanup needed when the pipeline shuts down.
        
    async def on_valves_updated(self):
        # This method is called by Open WebUI when the valve settings are changed in the UI.
        # The self.valves attribute should already be updated by the framework by the time this is called.
        print(f"on_valves_updated:{self.name} - Valves have been updated. Current configuration: {self.valves.model_dump_json(indent=2)}")
        # If there's any internal state that depends on valve values (e.g., pre-compiled templates),
        # it should be re-initialized here. For this pipeline, direct use of self.valves is sufficient.

    async def inlet(self, body: Dict[str, Any], user: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        print(f"inlet:{self.name} - Received request body: {json.dumps(body, indent=2)}")

        if "messages" not in body or not isinstance(body["messages"], list):
            print(f"inlet:{self.name} - 'messages' key missing or not a list in body. Skipping enhancement.")
            return body
        
        if not body["messages"]:
            print(f"inlet:{self.name} - 'messages' list is empty. Skipping enhancement.")
            return body

        # --- 1. Apply System Instruction ---
        # The system_instruction from valves will be used.
        # If a more specific guideline (like output_format_guideline) were active, it could be appended here:
        # effective_system_instruction = self.valves.system_instruction
        # if self.valves.output_format_guideline:
        #     if effective_system_instruction: # Check if it's not empty
        #         effective_system_instruction += f"\n\nOutput Formatting Guideline: {self.valves.output_format_guideline}"
        #     else: # If system_instruction was empty, this becomes the instruction
        #         effective_system_instruction = f"Output Formatting Guideline: {self.valves.output_format_guideline}"
        effective_system_instruction = self.valves.system_instruction

        if effective_system_instruction: # Only proceed if there's an instruction to apply
            # Check if the first message is already a system message
            if body["messages"][0].get("role") == "system":
                # Prepend to existing system message, avoiding duplication
                current_system_content = body["messages"][0]["content"]
                if not current_system_content.startswith(effective_system_instruction):
                    body["messages"][0]["content"] = f"{effective_system_instruction}\n{current_system_content}"
                    print(f"inlet:{self.name} - Prepended system instruction to existing system message.")
                else:
                    print(f"inlet:{self.name} - System instruction already present in the first system message.")
            else:
                # Insert new system message at the beginning
                body["messages"].insert(0, {"role": "system", "content": effective_system_instruction})
                print(f"inlet:{self.name} - Added new system instruction message at the beginning.")
        
        # --- 2. Apply User Prompt Prefix/Suffix ---
        # Find the last message with role "user" to modify its content.
        last_user_message_idx = -1
        for i in range(len(body["messages"]) - 1, -1, -1):
            if body["messages"][i].get("role") == "user":
                last_user_message_idx = i
                break
        
        if last_user_message_idx != -1:
            user_msg_obj = body["messages"][last_user_message_idx]
            
            # Handle string content
            if isinstance(user_msg_obj.get("content"), str):
                original_content = user_msg_obj["content"]
                modified_content = original_content

                if self.valves.user_prompt_prefix and not modified_content.startswith(self.valves.user_prompt_prefix):
                    modified_content = f"{self.valves.user_prompt_prefix}{modified_content}"
                
                if self.valves.user_prompt_suffix and not modified_content.endswith(self.valves.user_prompt_suffix):
                    modified_content = f"{modified_content}{self.valves.user_prompt_suffix}"
                
                if modified_content != original_content:
                    user_msg_obj["content"] = modified_content
                    print(f"inlet:{self.name} - Applied affixes to user message (string content). New content: '{modified_content}'")

            # Handle list content (for multimodal messages, apply to text parts)
            elif isinstance(user_msg_obj.get("content"), list):
                applied_to_list_content = False
                for part_idx, part in enumerate(user_msg_obj["content"]):
                    if isinstance(part, dict) and part.get("type") == "text":
                        original_text_part = part["text"]
                        modified_text_part = original_text_part

                        if self.valves.user_prompt_prefix and not modified_text_part.startswith(self.valves.user_prompt_prefix):
                            modified_text_part = f"{self.valves.user_prompt_prefix}{modified_text_part}"
                        
                        if self.valves.user_prompt_suffix and not modified_text_part.endswith(self.valves.user_prompt_suffix):
                            modified_text_part = f"{modified_text_part}{self.valves.user_prompt_suffix}"
                        
                        if modified_text_part != original_text_part:
                            user_msg_obj["content"][part_idx]["text"] = modified_text_part # Modify in place
                            applied_to_list_content = True
                
                if applied_to_list_content:
                     print(f"inlet:{self.name} - Applied affixes to user message (list content, text parts).")
            else:
                print(f"inlet:{self.name} - Last user message content is of an unsupported type for affixing: {type(user_msg_obj.get('content'))}.")
        else:
            print(f"inlet:{self.name} - No user message found in the conversation to apply prefix/suffix to.")

        print(f"inlet:{self.name} - Final enhanced body: {json.dumps(body, indent=2)}")
        return body

    async def outlet(self, body: Dict[str, Any], user: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # This filter primarily modifies the inlet. Outlet can be a pass-through.
        print(f"outlet:{self.name} - Passing through response body: {json.dumps(body, indent=2)}")
        return body
