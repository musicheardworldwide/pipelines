"""
title: SIN Knowledge RAG Pipe
author: Dr. Wes Caldwell & iA-SIN
description: Retrieves information from SIN's knowledge sources (Obsidian, Docs, KG) to augment generation.
version: 0.1.0
license: MIT
requirements: pydantic, llama-index, llama-index-llms-ollama, llama-index-embeddings-ollama # Example deps
"""

import os
import json
from typing import List, Union, Generator, Iterator, Dict, Any, Optional, AsyncGenerator

from pydantic import BaseModel, Field

# LlamaIndex imports (example)
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, QueryBundle
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.ollama import Ollama # Example LLM
from llama_index.embeddings.ollama import OllamaEmbedding # Example Embedding

class Pipeline:
    class Valves(BaseModel):
        rag_llm_model: str = Field(default=os.getenv("SIN_RAG_LLM_MODEL", "llama3"))
        rag_embedding_model: str = Field(default=os.getenv("SIN_RAG_EMBED_MODEL", "nomic-embed-text"))
        ollama_base_url: str = Field(default=os.getenv("SIN_OLLAMA_BASE_URL", "http://localhost:11434"))
        
        obsidian_vault_path: Optional[str] = Field(default=os.getenv("SIN_OBSIDIAN_VAULT_PATH", "./sin_data/obsidian_vault"))
        project_docs_path: Optional[str] = Field(default=os.getenv("SIN_PROJECT_DOCS_PATH", "./sin_data/project_docs"))
        # kg_access_endpoint: Optional[str] = Field(default=None) # For future KG direct access

        top_k_retrieval: int = Field(default=3)
        similarity_cutoff: float = Field(default=0.7)

    def __init__(self):
        self.type = "pipe" # This is a RAG pipe
        self.name = "SIN Knowledge RAG Pipe"
        self.valves = self.Valves()
        
        self.index = None # LlamaIndex VectorStoreIndex
        self.query_engine = None

        print(f"Initialized {self.name} with valves: {self.valves.model_dump_json(indent=2)}")

    async def on_startup(self):
        print(f"on_startup:{self.name}")
        
        Settings.llm = Ollama(model=self.valves.rag_llm_model, base_url=self.valves.ollama_base_url)
        Settings.embed_model = OllamaEmbedding(model_name=self.valves.rag_embedding_model, base_url=self.valves.ollama_base_url)

        documents = []
        if self.valves.obsidian_vault_path and os.path.exists(self.valves.obsidian_vault_path):
            print(f"{self.name}: Loading documents from Obsidian vault: {self.valves.obsidian_vault_path}")
            obsidian_docs = SimpleDirectoryReader(self.valves.obsidian_vault_path, recursive=True).load_data()
            documents.extend(obsidian_docs)
            print(f"{self.name}: Loaded {len(obsidian_docs)} documents from Obsidian.")

        if self.valves.project_docs_path and os.path.exists(self.valves.project_docs_path):
            print(f"{self.name}: Loading documents from project docs: {self.valves.project_docs_path}")
            project_docs = SimpleDirectoryReader(self.valves.project_docs_path, recursive=True).load_data()
            documents.extend(project_docs)
            print(f"{self.name}: Loaded {len(project_docs)} documents from project docs.")
        
        # Add KG loading here if/when available

        if not documents:
            print(f"{self.name}: No documents found in specified paths. RAG will be limited.")
            return

        print(f"{self.name}: Building VectorStoreIndex from {len(documents)} total documents...")
        self.index = VectorStoreIndex.from_documents(documents, show_progress=True)
        
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.valves.top_k_retrieval,
            # similarity_cutoff=self.valves.similarity_cutoff # Requires LlamaIndex >= 0.10.24
        )
        
        response_synthesizer = get_response_synthesizer(
            response_mode=ResponseMode.COMPACT, # Adjust as needed
            streaming=True 
        )
        
        # Assembling a basic query engine
        # For more complex RAG, you might use a QueryPipeline like in universal_rag.py
        self.query_engine = self.index.as_query_engine(
            streaming=True,
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            # similarity_top_k=self.valves.top_k_retrieval # Alternative way to set top_k
        )
        print(f"{self.name}: RAG Index and Query Engine initialized.")


    async def on_shutdown(self):
        print(f"on_shutdown:{self.name}")

    async def on_valves_updated(self):
        print(f"on_valves_updated:{self.name} - Valves updated. Re-initializing RAG setup...")
        # This would ideally re-run the relevant parts of on_startup
        # For simplicity, a full restart of the pipeline server might be easier if valves change often.
        await self.on_startup() 

    async def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> AsyncGenerator[str, None]:
        print(f"pipe:{self.name} - User query: {user_message}")

        if not self.query_engine:
            yield "SIN Knowledge RAG Engine is not initialized. Please check document paths and ensure data is loaded."
            return

        # Construct a query bundle if needed, or just pass the string
        query_bundle = QueryBundle(user_message)
        
        try:
            streaming_response = await self.query_engine.aquery(query_bundle)
            
            if hasattr(streaming_response, 'response_gen') and streaming_response.response_gen:
                # This is typical for LlamaIndex streaming responses
                async for token in streaming_response.response_gen:
                    yield token
                
                # Optionally, append source nodes information
                source_nodes_info = "\n\n--- Sources ---\n"
                for i, node in enumerate(streaming_response.source_nodes):
                    source_nodes_info += f"{i+1}. {node.metadata.get('file_name', 'Unknown Source')} (Score: {node.score:.2f})\n"
                yield source_nodes_info

            elif hasattr(streaming_response, 'response') and isinstance(streaming_response.response, str):
                 # Non-streaming response or if response_gen is not available
                yield streaming_response.response
                source_nodes_info = "\n\n--- Sources ---\n"
                for i, node in enumerate(streaming_response.source_nodes):
                    source_nodes_info += f"{i+1}. {node.metadata.get('file_name', 'Unknown Source')} (Score: {node.score:.2f})\n"
                yield source_nodes_info
            else:
                yield "Received an unexpected response structure from RAG query engine."

        except Exception as e:
            print(f"{self.name}: Error during RAG query: {e}")
            yield f"Error processing RAG query: {e}"