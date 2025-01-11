"""
title: Llama Index Ollama Github Filter Pipeline
author: Wes Caldwell
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library with Ollama embeddings from a GitHub repository.
requirements: llama-index, llama-index-llms-ollama, llama-index-embeddings-ollama, llama-index-readers-github
"""
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel, Field
import os
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Filter:
    """
    Llama Index Ollama GitHub Filter for Open WebUI.

    This filter retrieves relevant information from a GitHub repository using
    the Llama Index library with Ollama embeddings and provides it to the main model.
    """

    class Valves(BaseModel):
        """
        Configuration options for the GitHub filter.

        Attributes:
        -----------
        ENABLE_FILTER : bool
            Enable or disable the filter.
        GITHUB_TOKEN : str
            GitHub API token for accessing repositories.
        GITHUB_OWNER : str
            GitHub repository owner.
        GITHUB_REPO : str
            GitHub repository name.
        GITHUB_BRANCH : str
            Branch name to retrieve data from.
        """
        ENABLE_FILTER: bool = Field(
            default=True,
            description="Enable or disable the GitHub filter."
        )
        GITHUB_TOKEN: str = Field(
            default=os.environ.get("GITHUB_TOKEN", ""),
            description="GitHub API token for accessing repositories."
        )
        GITHUB_OWNER: str = Field(
            default="musicheardworldwide",
            description="GitHub repository owner."
        )
        GITHUB_REPO: str = Field(
            default="",
            description="GitHub repository name."
        )
        GITHUB_BRANCH: str = Field(
            default="main",
            description="Branch name to retrieve data from."
        )

    def __init__(self):
        """
        Initializes the filter with default or configured valves.
        """
        self.valves = self.Valves()
        self.documents = None
        self.index = None

    async def on_startup(self):
        """
        Initializes the Llama Index with GitHub data on startup.
        """
        if not self.valves.ENABLE_FILTER:
            logger.info("GitHub filter is disabled.")
            return

        try:
            from llama_index.embeddings.ollama import OllamaEmbedding
            from llama_index.llms.ollama import Ollama
            from llama_index.core import VectorStoreIndex, Settings
            from llama_index.readers.github import GithubRepositoryReader, GithubClient

            Settings.embed_model = OllamaEmbedding(
                model_name="text",
                base_url="https://api.musicheardworldwide.com",
            )
            Settings.llm = Ollama(model="llama3")

            github_client = GithubClient(
                github_token=self.valves.GITHUB_TOKEN, verbose=True
            )

            reader = GithubRepositoryReader(
                github_client=github_client,
                owner=self.valves.GITHUB_OWNER,
                repo=self.valves.GITHUB_REPO,
                use_parser=False,
                verbose=False,
                filter_file_extensions=(
                    [
                        ".png",
                        ".jpg",
                        ".jpeg",
                        ".gif",
                        ".svg",
                        ".ico",
                        "json",
                        ".ipynb",
                    ],
                    GithubRepositoryReader.FilterType.EXCLUDE,
                ),
            )

            loop = asyncio.new_event_loop()
            reader._loop = loop

            try:
                # Load data from the branch
                self.documents = await asyncio.to_thread(
                    reader.load_data, branch=self.valves.GITHUB_BRANCH
                )
                self.index = VectorStoreIndex.from_documents(self.documents)
                logger.info(f"Loaded {len(self.documents)} documents into the index.")
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Error initializing GitHub filter: {e}")

    async def on_shutdown(self):
        """
        Cleans up resources on shutdown.
        """
        logger.info("GitHub filter is shutting down.")

    def retrieve_github_knowledge(self, user_message: str) -> Union[str, Generator]:
        """
        Queries the Llama Index for relevant information.

        Parameters:
        ----------
        user_message : str
            User's query message.

        Returns:
        -------
        str or Generator
            Response from the Llama Index query engine.
        """
        if not self.index:
            return "GitHub index is not initialized. Unable to retrieve knowledge."

        query_engine = self.index.as_query_engine(streaming=True)
        response = query_engine.query(user_message)
        return response.response_gen

    def inlet(self, body: dict, **kwargs) -> dict:
        """
        Pre-processes user input (no changes in this implementation).

        Parameters:
        ----------
        body : dict
            Input body containing user messages.

        Returns:
        -------
        dict
            Unmodified input body.
        """
        return body

    def outlet(self, body: dict, **kwargs) -> dict:
        """
        Processes the output by retrieving relevant GitHub knowledge.

        Parameters:
        ----------
        body : dict
            Outgoing response body.

        Returns:
        -------
        dict
            Modified response body with GitHub knowledge appended.
        """
        if not self.valves.ENABLE_FILTER:
            return body

        user_message = body.get("messages", [])[-1].get("content", "")
        if not user_message:
            logger.warning("No user message found.")
            return body

        # Retrieve GitHub knowledge
        github_response = self.retrieve_github_knowledge(user_message)

        # Append GitHub knowledge to the assistant's response
        body["messages"].append({
            "role": "assistant",
            "content": "Here is some relevant knowledge from the GitHub repository:\n" + "".join(github_response)
        })

        return body

# Example Usage
if __name__ == "__main__":
    filter_instance = Filter()

    # Simulate startup
    asyncio.run(filter_instance.on_startup())

    # Simulated user prompt
    input_body = {
        "messages": [
            {"role": "user", "content": "Explain the CI/CD pipeline in this repository."}
        ]
    }

    # Process the prompt through the filter
    output_body = filter_instance.outlet(input_body)
    print("Processed Output:", output_body)

    # Simulate shutdown
    asyncio.run(filter_instance.on_shutdown())
