"""
title: SIN Knowledge RAG Pipe
author: Dr. Wes Caldwell & iA-SIN
description: Retrieves information from SIN's knowledge sources (Obsidian, Docs, KG) to augment generation.
version: 0.1.1
license: MIT
requirements: pydantic, llama-index>=0.10.24, llama-index-llms-ollama, llama-index-embeddings-ollama, llama-index-readers-file
"""

import os
import json
import logging # Added for better logging
from typing import List, Union, Generator, Iterator, Dict, Any, Optional, AsyncGenerator

from pydantic import BaseModel, Field

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, QueryBundle
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

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
        rag_llm_model: str = Field(default=os.getenv("SIN_RAG_LLM_MODEL", "llama3"))
        rag_embedding_model: str = Field(default=os.getenv("SIN_RAG_EMBED_MODEL", "nomic-embed-text"))
        ollama_base_url: str = Field(default=os.getenv("SIN_OLLAMA_BASE_URL", "http://docker.host.internal:11434"))
        
        # Ensure these paths are accessible from where this pipeline code runs (e.g., inside Docker container if applicable)
        obsidian_vault_path: Optional[str] = Field(default=os.getenv("SIN_OBSIDIAN_VAULT_PATH", "/app/mounted_data/obsidian_vault")) # Example Docker path
        project_docs_path: Optional[str] = Field(default=os.getenv("SIN_PROJECT_DOCS_PATH", "/app/mounted_data/project_docs")) # Example Docker path
        
        top_k_retrieval: int = Field(default=3)
        similarity_cutoff: float = Field(default=0.7) # Requires LlamaIndex >= 0.10.24

    def __init__(self):
        self.type = "pipe"
        self.name = "SIN Knowledge RAG Pipe"
        self.valves = self.Valves()
        
        self.index = None
        self.query_engine = None

        logger.info(f"Initialized {self.name} with valves: {self.valves.model_dump_json(indent=2)}")

    async def on_startup(self):
        logger.info(f"on_startup:{self.name} - Starting initialization...")
        
        try:
            logger.info(f"Setting LLM: model={self.valves.rag_llm_model}, base_url={self.valves.ollama_base_url}")
            Settings.llm = Ollama(model=self.valves.rag_llm_model, base_url=self.valves.ollama_base_url, request_timeout=120.0) # Added timeout
            
            logger.info(f"Setting Embed Model: model_name={self.valves.rag_embedding_model}, base_url={self.valves.ollama_base_url}")
            Settings.embed_model = OllamaEmbedding(model_name=self.valves.rag_embedding_model, base_url=self.valves.ollama_base_url)
            logger.info("LlamaIndex Settings configured.")

        except Exception as e:
            logger.error(f"Error configuring LlamaIndex Settings (LLM/Embed Model): {e}", exc_info=True)
            # Attempt to ping Ollama to see if it's reachable
            try:
                import httpx
                ping_url = f"{self.valves.ollama_base_url.rstrip('/')}/api/tags"
                logger.info(f"Attempting to ping Ollama at {ping_url}...")
                response = httpx.get(ping_url, timeout=5.0)
                logger.info(f"Ollama ping response status: {response.status_code}")
                if response.status_code != 200:
                    logger.error(f"Ollama ping failed. Response: {response.text}")
                else:
                    logger.info(f"Ollama ping successful. Models available: {response.json().get('models', [])}")
            except Exception as ping_e:
                logger.error(f"Failed to ping Ollama at {self.valves.ollama_base_url}: {ping_e}", exc_info=True)
            return # Stop further initialization if settings fail

        documents = []
        # --- Obsidian Path ---
        obs_path = self.valves.obsidian_vault_path
        logger.info(f"Checking Obsidian vault path: {obs_path}")
        if obs_path and os.path.exists(obs_path):
            if os.path.isdir(obs_path):
                logger.info(f"Loading documents from Obsidian vault: {obs_path}")
                try:
                    obsidian_docs = SimpleDirectoryReader(obs_path, recursive=True).load_data()
                    documents.extend(obsidian_docs)
                    logger.info(f"Loaded {len(obsidian_docs)} documents from Obsidian.")
                except Exception as e:
                    logger.error(f"Error loading documents from Obsidian path '{obs_path}': {e}", exc_info=True)
            else:
                logger.warning(f"Obsidian path '{obs_path}' is not a directory.")
        elif obs_path:
            logger.warning(f"Obsidian path '{obs_path}' does not exist.")
        else:
            logger.info("Obsidian path not configured.")

        # --- Project Docs Path ---
        proj_path = self.valves.project_docs_path
        logger.info(f"Checking project docs path: {proj_path}")
        if proj_path and os.path.exists(proj_path):
            if os.path.isdir(proj_path):
                logger.info(f"Loading documents from project docs: {proj_path}")
                try:
                    project_docs = SimpleDirectoryReader(proj_path, recursive=True).load_data()
                    documents.extend(project_docs)
                    logger.info(f"Loaded {len(project_docs)} documents from project docs.")
                except Exception as e:
                    logger.error(f"Error loading documents from project docs path '{proj_path}': {e}", exc_info=True)
            else:
                logger.warning(f"Project docs path '{proj_path}' is not a directory.")
        elif proj_path:
            logger.warning(f"Project docs path '{proj_path}' does not exist.")
        else:
            logger.info("Project docs path not configured.")
        
        if not documents:
            logger.warning(f"{self.name}: No documents found in specified paths. RAG functionality will be limited or unavailable.")
            return # Don't proceed to build index if no documents

        logger.info(f"{self.name}: Building VectorStoreIndex from {len(documents)} total documents...")
        try:
            self.index = VectorStoreIndex.from_documents(documents, show_progress=True)
            logger.info(f"VectorStoreIndex built successfully.")
        except Exception as e:
            logger.error(f"Error building VectorStoreIndex: {e}", exc_info=True)
            return

        try:
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=self.valves.top_k_retrieval,
            )
            # If LlamaIndex version is new enough, you can add similarity_cutoff
            # from llama_index import __version__ as llama_version
            # from packaging.version import parse as parse_version
            # if parse_version(llama_version) >= parse_version("0.10.24"):
            #    retriever.similarity_cutoff = self.valves.similarity_cutoff
            
            response_synthesizer = get_response_synthesizer(
                response_mode=ResponseMode.COMPACT, 
                streaming=True,
                llm=Settings.llm # Explicitly pass the LLM
            )
            
            self.query_engine = self.index.as_query_engine(
                streaming=True,
                retriever=retriever,
                response_synthesizer=response_synthesizer,
            )
            logger.info(f"{self.name}: RAG Index and Query Engine initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing Query Engine components: {e}", exc_info=True)


    async def on_shutdown(self):
        logger.info(f"on_shutdown:{self.name}")

    async def on_valves_updated(self):
        logger.info(f"on_valves_updated:{self.name} - Valves updated. Re-initializing RAG setup...")
        # Reset state before re-initializing
        self.index = None
        self.query_engine = None
        Settings.llm = None # Reset to allow re-configuration
        Settings.embed_model = None # Reset to allow re-configuration
        await self.on_startup() 

    async def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> AsyncGenerator[str, None]:
        logger.info(f"pipe:{self.name} - User query: '{user_message}' for model_id: '{model_id}'")

        if not self.query_engine:
            logger.warning("Query engine not initialized. Returning warning message.")
            yield "SIN Knowledge RAG Engine is not initialized. Please check server logs for errors, document paths, and ensure data is loaded."
            return
        if not self.index:
            logger.warning("Index not initialized. Returning warning message.")
            yield "SIN Knowledge RAG Index is not built. Please check server logs for errors during document loading or index creation."
            return


        query_bundle = QueryBundle(user_message)
        
        try:
            logger.info(f"Querying RAG engine with: '{user_message}'")
            # Ensure aquery is used for async operation
            streaming_response = await self.query_engine.aquery(query_bundle) 
            
            if hasattr(streaming_response, 'response_gen') and streaming_response.response_gen:
                logger.info("Processing response_gen stream...")
                chunk_count = 0
                async for token in streaming_response.response_gen:
                    yield token
                    chunk_count +=1
                logger.info(f"Streamed {chunk_count} tokens from response_gen.")
                
                source_nodes_info = "\n\n--- Sources ---\n"
                if hasattr(streaming_response, 'source_nodes') and streaming_response.source_nodes:
                    for i, node in enumerate(streaming_response.source_nodes):
                        file_name = node.metadata.get('file_name', 'Unknown Source')
                        # Truncate long file names if necessary
                        if len(file_name) > 70: 
                            file_name = "..." + file_name[-67:]
                        source_nodes_info += f"{i+1}. {file_name} (Score: {node.score:.2f})\n"
                    logger.info(f"Appending source nodes: {source_nodes_info.strip()}")
                    yield source_nodes_info
                else:
                    logger.info("No source nodes found in streaming_response.")

            elif hasattr(streaming_response, 'response') and isinstance(streaming_response.response, str):
                logger.info(f"Processing non-streaming response: {streaming_response.response[:100]}...") # Log first 100 chars
                yield streaming_response.response
                
                source_nodes_info = "\n\n--- Sources ---\n"
                if hasattr(streaming_response, 'source_nodes') and streaming_response.source_nodes:
                    for i, node in enumerate(streaming_response.source_nodes):
                        file_name = node.metadata.get('file_name', 'Unknown Source')
                        if len(file_name) > 70:
                             file_name = "..." + file_name[-67:]
                        source_nodes_info += f"{i+1}. {file_name} (Score: {node.score:.2f})\n"
                    logger.info(f"Appending source nodes: {source_nodes_info.strip()}")
                    yield source_nodes_info
                else:
                    logger.info("No source nodes found in non-streaming response object.")
            else:
                logger.warning(f"Received an unexpected response structure from RAG query engine: {type(streaming_response)}")
                yield "Received an unexpected response structure from RAG query engine."

        except Exception as e:
            logger.error(f"Error during RAG query execution: {e}", exc_info=True)
            yield f"Error processing RAG query: {str(e)}"
