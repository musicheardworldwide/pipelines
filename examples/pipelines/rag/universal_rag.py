"""
LlamaIndex RAG Pipeline Template for Open WebUI

This pipeline connects LlamaIndex's retrieval-augmented generation (RAG)
features with Open WebUI's pipe function support. It includes a full stack:
prompt rewriting, document retrieval, reranking, and summarization using LLMs.

Supported components:
- OpenAI LLM (chat/completion)
- Cohere Reranker (optional)
- Vector-based retriever from local documents
"""

import os
from typing import Dict, List, Optional, Union
from pydantic import Field
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    QueryPipeline,
)
from llama_index.llms import OpenAI
from llama_index.prompts import PromptTemplate
from llama_index.postprocessor import CohereRerank
from llama_index.response_synthesizers import TreeSummarize
from llama_index.query_pipeline import InputComponent
from blueprints.function_calling_blueprint import Pipeline as FunctionCallingBlueprint


class Pipeline(FunctionCallingBlueprint):
    """
    RAG pipeline for document-grounded chat in Open WebUI using LlamaIndex.

    Combines prompt transformation, retrieval, reranking, and summarization.
    Automatically reads documents from a local folder and builds a retriever index.
    """

    class Valves(FunctionCallingBlueprint.Valves):
        """
        Valve configuration for the RAG pipeline.

        Attributes:
            OPENAI_API_KEY (str): OpenAI API key for LLM access.
            COHERE_API_KEY (str): Cohere API key for reranking (optional).
            DATA_DIR (str): Directory path containing the source documents.
            OPENAI_MODEL (str): LLM model name (e.g., "gpt-3.5-turbo").
            TOP_K (int): Number of top documents to retrieve from the vector index.
        """
        OPENAI_API_KEY: str = Field(default="", description="OpenAI API Key")
        COHERE_API_KEY: str = Field(default="", description="Cohere API Key")
        DATA_DIR: str = Field(default="./data", description="Path to documents for indexing")
        OPENAI_MODEL: str = Field(default="gpt-3.5-turbo", description="OpenAI model to use")
        TOP_K: int = Field(default=5, description="Top K documents for retrieval")

    def __init__(self):
        """
        Initialize the pipeline with environment variables and construct the RAG pipeline.
        """
        super().__init__()
        self.name = "LlamaIndex RAG Pipeline"
        self.type = "pipe"

        # Inject env vars into Valves config
        self.valves = self.Valves(**{
            **self.valves.model_dump(),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
            "COHERE_API_KEY": os.getenv("COHERE_API_KEY", "")
        })

        # Create the pipeline graph
        self.pipeline = self._init_query_pipeline()

    def _init_query_pipeline(self) -> QueryPipeline:
        """
        Build the internal LlamaIndex query pipeline.

        Returns:
            QueryPipeline: Configured LlamaIndex pipeline with custom components.
        """
        llm = OpenAI(model=self.valves.OPENAI_MODEL, api_key=self.valves.OPENAI_API_KEY)

        # Index from local documents
        documents = SimpleDirectoryReader(self.valves.DATA_DIR).load_data()
        index = VectorStoreIndex.from_documents(documents)
        retriever = index.as_retriever(similarity_top_k=self.valves.TOP_K)

        # Build pipeline nodes
        prompt = PromptTemplate("Please generate a concise question about {topic}")
        reranker = CohereRerank(api_key=self.valves.COHERE_API_KEY)
        summarizer = TreeSummarize(service_context=ServiceContext.from_defaults(llm=llm))

        # Assemble the query pipeline
        pipeline = QueryPipeline(verbose=True)
        pipeline.add_modules({
            "input": InputComponent(),
            "prompt_tmpl": prompt,
            "llm": llm,
            "retriever": retriever,
            "reranker": reranker,
            "summarizer": summarizer,
        })

        # Define data flow links
        pipeline.add_link("input", "prompt_tmpl", dest_key="topic")
        pipeline.add_link("prompt_tmpl", "llm")
        pipeline.add_link("llm", "retriever")
        pipeline.add_link("retriever", "reranker", dest_key="nodes")
        pipeline.add_link("llm", "reranker", dest_key="query_str")
        pipeline.add_link("reranker", "summarizer", dest_key="nodes")
        pipeline.add_link("llm", "summarizer", dest_key="query_str")

        return pipeline

    async def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[Dict[str, Union[str, Dict]]],
        body: Dict[str, Union[str, bool]],
    ) -> str:
        """
        Open WebUI interface method. Executes the configured RAG pipeline.

        Args:
            user_message (str): Raw user input message.
            model_id (str): Model name (ignored, configured via Valves).
            messages (List[Dict]): Optional chat history context.
            body (Dict): Additional metadata or stream flag.

        Returns:
            str: Final LLM-generated response string from RAG process.
        """
        response = self.pipeline.run(input=user_message)
        return str(response)
