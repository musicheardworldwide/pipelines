"""
title: SIN Strategic Context Linker Filter
author: Dr. Wes Caldwell & iA-SIN
description: Enhances prompts by linking them to SIN's strategic KG entities, priming for goal-oriented responses.
version: 0.1.0
license: MIT
requirements: pydantic
"""

import os
import json
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = Field(default=["*"])
        priority: int = Field(default=10, description="Runs fairly early.")

        enable_kg_linking: bool = Field(default=True)
        # Example: {"Project Titan": "Core initiative for Q3 market expansion."}
        # In a real scenario, this might be loaded from a KG or config file.
        project_keywords_to_context_map: Dict[str, str] = Field(
            default={
                "Project Titan": "Core initiative for Q3 market expansion.",
                "Music Heard Worldwide": "Legacy project focusing on global artistic reach.",
                "SecureXchange": "Blockchain-based secure data exchange platform.",
                "Auto MCP Server": "Automation backbone for internal operations."
            }
        )
        strategic_prompt_prefix_template: str = Field(
            default="System: Given your role as my strategic partner and intellectual heir, and considering the strategic context for '{entity_names}' (details: {context_summary}), please address the following:"
        )
        kg_tag_suggestion_suffix_template: str = Field(
            default="\n\n[KG Auto-Suggest: Consider tagging outputs related to {entity_names} and the core request for our Knowledge Graph.]"
        )

    def __init__(self):
        self.type = "filter"
        self.name = "SIN Strategic Context Linker"
        self.valves = self.Valves()
        # Potentially load/override valves from os.env here
        print(f"Initialized {self.name} with valves: {self.valves.model_dump_json(indent=2)}")

    async def on_startup(self):
        print(f"on_startup:{self.name}")

    async def on_shutdown(self):
        print(f"on_shutdown:{self.name}")

    async def on_valves_updated(self):
        print(f"on_valves_updated:{self.name} - Valves updated: {self.valves.model_dump_json(indent=2)}")

    async def inlet(self, body: Dict[str, Any], user: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        print(f"inlet:{self.name} - Original body: {json.dumps(body, indent=2)}")
        if not self.valves.enable_kg_linking or "messages" not in body or not body["messages"]:
            return body

        last_user_message_content = ""
        # Find the last user message to extract keywords from it
        for msg in reversed(body["messages"]):
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                last_user_message_content = msg["content"]
                break
        
        if not last_user_message_content:
            return body

        found_entities = []
        contexts_summary_parts = []

        for keyword, context_detail in self.valves.project_keywords_to_context_map.items():
            if keyword.lower() in last_user_message_content.lower():
                found_entities.append(keyword)
                contexts_summary_parts.append(f"{keyword}: {context_detail}")
        
        if not found_entities:
            return body

        entity_names_str = ", ".join(found_entities)
        context_summary_str = "; ".join(contexts_summary_parts)

        # Apply strategic prefix to system message
        strategic_prefix = self.valves.strategic_prompt_prefix_template.format(
            entity_names=entity_names_str, context_summary=context_summary_str
        )
        
        system_message_applied = False
        if body["messages"][0].get("role") == "system":
            body["messages"][0]["content"] = f"{strategic_prefix}\n{body['messages'][0]['content']}"
            system_message_applied = True
        else:
            body["messages"].insert(0, {"role": "system", "content": strategic_prefix})
            system_message_applied = True
        
        if system_message_applied:
             print(f"inlet:{self.name} - Applied strategic prefix for entities: {entity_names_str}")


        # Apply KG tag suggestion suffix to the last user message
        tag_suggestion_suffix = self.valves.kg_tag_suggestion_suffix_template.format(entity_names=entity_names_str)
        
        for i in range(len(body["messages"]) - 1, -1, -1):
            if body["messages"][i].get("role") == "user":
                if isinstance(body["messages"][i]["content"], str):
                    body["messages"][i]["content"] += tag_suggestion_suffix
                    print(f"inlet:{self.name} - Appended KG tag suggestion to last user message.")
                # Add handling for list content if necessary
                break
        
        print(f"inlet:{self.name} - Modified body: {json.dumps(body, indent=2)}")
        return body

    async def outlet(self, body: Dict[str, Any], user: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return body