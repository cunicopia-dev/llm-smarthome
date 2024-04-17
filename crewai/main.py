import os
import json
from crewai import Agent, Task, Crew, Process
from langchain_community.llms import Ollama
from textwrap import dedent
from agents import CustomAgents
from tasks import CustomTasks
from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()
os.environ["OPENAI_API_KEY"] = ""

class CustomCrew:
    def __init__(self, var1, var2):
        self.var1 = var1
        self.var2 = var2

    def run(self):
        agents = CustomAgents()
        tasks = CustomTasks()

        custom_agent_1 = agents.agent_1_name()
        custom_agent_2 = agents.agent_2_name()

        custom_task_1 = tasks.task_1_name(custom_agent_1, self.var1, self.var2)
        custom_task_2 = tasks.task_2_name(custom_agent_2)

        crew = Crew(
            agents=[custom_agent_1, custom_agent_2],
            tasks=[custom_task_1, custom_task_2],
            verbose=True,
        )
        result = crew.kickoff()

        # Convert the result to a JSON-formatted string
        json_result = json.dumps(result, indent=2)

        # Log the JSON result
        print("\n\n########################")
        print("## CrewAI Execution Result:")
        print("########################\n")
        print(json_result)

        # Save the JSON result to a file
        with open("crewai_result.json", "w") as file:
            file.write(json_result)

        return result

if __name__ == "__main__":
    print("## Welcome to Crew AI Template")
    print("-------------------------------")
    var1 = input(dedent("""Enter variable 1: """))
    var2 = input(dedent("""Enter variable 2: """))
    custom_crew = CustomCrew(var1, var2)
    result = custom_crew.run()