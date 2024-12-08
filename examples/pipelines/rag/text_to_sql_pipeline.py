"""
title: Llama Index DB Pipeline
author: 0xThresh
date: 2024-08-11
version: 1.1
license: MIT
description: A pipeline for using text-to-SQL for retrieving relevant information from a database using the Llama Index library.
requirements: llama_index, sqlalchemy, psycopg2-binary
"""

from typing import List, Union, Generator, Iterator
import os
from pydantic import BaseModel
from llama_index.llms.ollama import Ollama
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core import SQLDatabase, PromptTemplate
from sqlalchemy import create_engine

class Pipeline:
    class Valves(BaseModel):
        DB_HOST: str
        DB_PORT: str
        DB_USER: str
        DB_PASSWORD: str        
        DB_DATABASE: str
        DB_TABLE: str
        OLLAMA_HOST: str
        TEXT_TO_SQL_MODEL: str 

    def __init__(self):
        self.name = "Database RAG Pipeline"
        self.engine = None
        self.nlsql_response = ""

        # Initialize
        self.valves = self.Valves(
            **{
                "DB_HOST": os.getenv("DB_HOST", "localhost"),  # Plain host
                "DB_PORT": os.getenv("DB_PORT", "3306"),       # Default Postgres port
                "DB_USER": os.getenv("DB_USER", "sin"),       # Update as needed
                "DB_PASSWORD": os.getenv("DB_PASSWORD", "WORLDOFSIN1!"),  # Update as needed
                "DB_DATABASE": os.getenv("DB_DATABASE", "mhw_db"),    # Update as needed
                "DB_TABLE": os.getenv("DB_TABLE", "users"),           # Table(s) to run queries against 
                "OLLAMA_HOST": os.getenv("OLLAMA_HOST", "https://api.musicheardworldwide.com"),  # Local or remote Ollama host
                "TEXT_TO_SQL_MODEL": os.getenv("TEXT_TO_SQL_MODEL", "sin:latest")   # Model to use for text-to-SQL generation      
            }
        )

    def init_db_connection(self):
        try:
            # Initialize DB connection
            self.engine = create_engine(
                f"postgresql+psycopg2://{self.valves.DB_USER}:{self.valves.DB_PASSWORD}"
                f"@{self.valves.DB_HOST}:{self.valves.DB_PORT}/{self.valves.DB_DATABASE}"
            )
            return self.engine
        except Exception as e:
            raise RuntimeError(f"Database connection failed: {e}")

    async def on_startup(self):
        # Called on server startup
        self.init_db_connection()

    async def on_shutdown(self):
        # Called on server shutdown
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        try:
            # Initialize the database and LLM connections
            sql_database = SQLDatabase(self.engine, include_tables=[self.valves.DB_TABLE])
            llm = Ollama(
                model=self.valves.TEXT_TO_SQL_MODEL,
                base_url=self.valves.OLLAMA_HOST,
                request_timeout=180.0,
                context_window=30000
            )

            # Define the SQL generation prompt template
            text_to_sql_prompt = """
            Given an input question, first create a syntactically correct {dialect} query to run, 
            then look at the results of the query and return the answer. Query up to 5 rows using LIMIT.

            {schema}

            Question: {query_str}
            SQLQuery:
            """
            text_to_sql_template = PromptTemplate(
                text=text_to_sql_prompt,
                input_variables=["dialect", "schema", "query_str"]
            )

            # Create the query engine
            query_engine = NLSQLTableQueryEngine(
                sql_database=sql_database,
                tables=[self.valves.DB_TABLE],
                llm=llm,
                text_to_sql_prompt=text_to_sql_template,
                streaming=True
            )

            # Execute the query
            response = query_engine.query(user_message)

            # Return concatenated response from the generator
            return "".join(response.response_gen)

        except Exception as e:
            return f"Pipeline error: {e}"

# Example usage
if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.init_db_connection()  # Ensure the connection works

    # Simulate a user query
    user_query = "What are the top 5 users with the most posts?"
    response = pipeline.pipe(
        user_message=user_query,
        model_id="sin:latest",
        messages=[],
        body={}
    )
    print(response)
