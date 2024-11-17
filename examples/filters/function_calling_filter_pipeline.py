import os
import requests
from typing import Literal, List, Optional
from datetime import datetime

from blueprints.function_calling_blueprint import Pipeline as FunctionCallingBlueprint


class Pipeline(FunctionCallingBlueprint):
    class Valves(FunctionCallingBlueprint.Valves):
        # Add your custom parameters here
        OPENWEATHERMAP_API_KEY: str = ""
        pass

    class Tools:
        def __init__(self, pipeline) -> None:
            self.pipeline = pipeline

        def get_current_time(self) -> str:
            """
            Get the current time.

            :return: The current time.
            """
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            return f"Current Time = {current_time}"

        def get_current_weather(
            self,
            location: str,
            unit: Literal["metric", "fahrenheit"] = "fahrenheit",
        ) -> str:
            """
            Get the current weather for a location. If the location is not found, return an empty string.

            :param location: The location to get the weather for.
            :param unit: The unit to get the weather in. Default is fahrenheit.
            :return: The current weather for the location.
            """
            if self.pipeline.valves.OPENWEATHERMAP_API_KEY == "":
                return "OpenWeatherMap API Key not set, ask the user to set it up."
            else:
                units = "imperial" if unit == "fahrenheit" else "metric"
                params = {
                    "q": location,
                    "appid": self.pipeline.valves.OPENWEATHERMAP_API_KEY,
                    "units": units,
                }

                response = requests.get(
                    "http://api.openweathermap.org/data/2.5/weather", params=params
                )
                response.raise_for_status()  # Raises an HTTPError for bad responses
                data = response.json()

                weather_description = data["weather"][0]["description"]
                temperature = data["main"]["temp"]

                return f"{location}: {weather_description.capitalize()}, {temperature}°{unit.capitalize()[0]}"

        def calculate(self, equation: str) -> str:
            """
            Calculate the result of an equation.

            :param equation: The equation to calculate.
            """
            try:
                result = eval(equation)
                return f"{equation} = {result}"
            except Exception as e:
                print(e)
                return "Invalid equation"

        def get_stock_price(self, ticker: str) -> str:
            """
            Get the current stock price for a given ticker symbol.

            :param ticker: The stock ticker symbol.
            :return: The current stock price.
            """
            # Placeholder implementation
            return f"Stock price for {ticker}: $100.00"

        def get_news(self, query: str) -> str:
            """
            Get the latest news articles based on a query.

            :param query: The query to search for news.
            :return: The latest news articles.
            """
            # Placeholder implementation
            return f"Latest news for {query}: [Article 1, Article 2]"

        def get_wikipedia_summary(self, topic: str) -> str:
            """
            Get a summary from Wikipedia for a given topic.

            :param topic: The topic to search for.
            :return: The Wikipedia summary.
            """
            # Placeholder implementation
            return f"Wikipedia summary for {topic}: [Summary]"

        def convert_units(self, value: float, from_unit: str, to_unit: str) -> str:
            """
            Convert a value from one unit to another.

            :param value: The value to convert.
            :param from_unit: The unit to convert from.
            :param to_unit: The unit to convert to.
            :return: The converted value.
            """
            # Placeholder implementation
            return f"{value} {from_unit} = {value * 2} {to_unit}"

        def format_text(self, text: str, format: Literal["markdown", "html", "plain"]) -> str:
            """
            Format text into a specified format.

            :param text: The text to format.
            :param format: The format to convert the text to.
            :return: The formatted text.
            """
            # Placeholder implementation
            return f"Formatted text in {format}: {text}"

        def create_directory(self, path: str) -> str:
            """
            Create a directory at the specified path.

            :param path: The path where the directory should be created.
            :return: A message indicating the success or failure of the operation.
            """
            try:
                os.makedirs(path, exist_ok=True)
                return f"Directory created successfully at {path}"
            except Exception as e:
                return f"Error: Unable to create directory. {str(e)}"

        def delete_directory(self, path: str) -> str:
            """
            Delete a directory at the specified path.

            :param path: The path of the directory to be deleted.
            :return: A message indicating the success or failure of the operation.
            """
            try:
                os.rmdir(path)
                return f"Directory deleted successfully at {path}"
            except Exception as e:
                return f"Error: Unable to delete directory. {str(e)}"

        def list_directory_contents(self, path: str) -> str:
            """
            List the contents of a directory at the specified path.

            :param path: The path of the directory whose contents should be listed.
            :return: A string listing the contents of the directory.
            """
            try:
                contents = os.listdir(path)
                return f"Contents of directory {path}: {', '.join(contents)}"
            except Exception as e:
                return f"Error: Unable to list directory contents. {str(e)}"

        def download_file(self, url: str, save_path: str) -> str:
            """
            Download a file from the given URL and save it to the specified path.

            :param url: The URL of the file to be downloaded.
            :param save_path: The path where the file should be saved.
            :return: A message indicating the success or failure of the operation.
            """
            try:
                response = requests.get(url)
                with open(save_path, "wb") as file:
                    file.write(response.content)
                return f"File downloaded successfully and saved at {save_path}"
            except Exception as e:
                return f"Error: Unable to download file. {str(e)}"

        def send_email(self, to: str, subject: str, body: str) -> str:
            """
            Send an email.

            :param to: The recipient's email address.
            :param subject: The subject of the email.
            :param body: The body of the email.
            :return: A message indicating the success or failure of the operation.
            """
            # Placeholder implementation
            return f"Email sent to {to} with subject '{subject}'"

        def post_to_webhook(self, url: str, data: dict) -> str:
            """
            Post data to a webhook.

            :param url: The URL of the webhook.
            :param data: The data to post.
            :return: A message indicating the success or failure of the operation.
            """
            try:
                response = requests.post(url, json=data)
                response.raise_for_status()
                return f"Data posted successfully to {url}"
            except Exception as e:
                return f"Error: Unable to post data to webhook. {str(e)}"

        def scrape_webpage(self, url: str) -> str:
            """
            Scrape content from a webpage.

            :param url: The URL of the webpage to scrape.
            :return: The scraped content.
            """
            try:
                response = requests.get(url)
                response.raise_for_status()
                return f"Scraped content from {url}: {response.text}"
            except Exception as e:
                return f"Error: Unable to scrape webpage. {str(e)}"

        def query_database(self, query: str) -> str:
            """
            Execute a SQL query against a database.

            :param query: The SQL query to execute.
            :return: The result of the query.
            """
            # Placeholder implementation
            return f"Query result: [Result]"

        def insert_data(self, table: str, data: dict) -> str:
            """
            Insert data into a database table.

            :param table: The table to insert data into.
            :param data: The data to insert.
            :return: A message indicating the success or failure of the operation.
            """
            # Placeholder implementation
            return f"Data inserted into table {table}"

        def update_data(self, table: str, data: dict, condition: str) -> str:
            """
            Update data in a database table based on a condition.

            :param table: The table to update data in.
            :param data: The data to update.
            :param condition: The condition for the update.
            :return: A message indicating the success or failure of the operation.
            """
            # Placeholder implementation
            return f"Data updated in table {table} with condition {condition}"

        def generate_random_number(self, min: int, max: int) -> str:
            """
            Generate a random number within a specified range.

            :param min: The minimum value of the range.
            :param max: The maximum value of the range.
            :return: The generated random number.
            """
            import random
            return f"Random number: {random.randint(min, max)}"

        def generate_password(self, length: int) -> str:
            """
            Generate a random password of a specified length.

            :param length: The length of the password.
            :return: The generated password.
            """
            import string
            import random
            characters = string.ascii_letters + string.digits + string.punctuation
            password = ''.join(random.choice(characters) for i in range(length))
            return f"Generated password: {password}"

        def translate_text(self, text: str, target_language: str) -> str:
            """
            Translate text into a specified language.

            :param text: The text to translate.
            :param target_language: The language to translate the text to.
            :return: The translated text.
            """
            # Placeholder implementation
            return f"Translated text to {target_language}: {text}"

        def run_python_code(self, code: str) -> str:
            """
            Execute a Python code snippet.

            :param code: The Python code to execute.
            :return: The output of the code.
            """
            import io
            import contextlib
            import sys
            try:
                old_stdout = sys.stdout
                redirected_output = io.StringIO()
                sys.stdout = redirected_output
                exec(code)
                sys.stdout = old_stdout
                return redirected_output.getvalue()
            except Exception as e:
                return f"Error: {str(e)}"

        def run_shell_command(self, command: str) -> str:
            """
            Execute a shell command.

            :param command: The shell command to execute.
            :return: The output of the command.
            """
            import subprocess
            try:
                result = subprocess.run(command, shell=True, capture_output=True, text=True)
                return result.stdout
            except Exception as e:
                return f"Error: {str(e)}"

        def schedule_task(self, task: str, time: str) -> str:
            """
            Schedule a task to run at a specified time.

            :param task: The task to schedule.
            :param time: The time to run the task.
            :return: A message indicating the success or failure of the operation.
            """
            # Placeholder implementation
            return f"Task '{task}' scheduled to run at {time}"

    def __init__(self):
        super().__init__()
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "my_tools_pipeline"
        self.name = "My Tools Pipeline"
        self.valves = self.Valves(
            **{
                **self.valves.model_dump(),
                "pipelines": ["*"],  # Connect to all pipelines
                "OPENWEATHERMAP_API_KEY": os.getenv("OPENWEATHERMAP_API_KEY", ""),
            },
        )
        self.tools = self.Tools(self)
