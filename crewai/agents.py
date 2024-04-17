from crewai import Agent
from textwrap import dedent
from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()

class CustomAgents:
    def __init__(self):
        self.ollama_model = "mistral:7b-instruct-v0.2-fp16"
        self.Ollama = Ollama(model=self.ollama_model)

    def information_manager(self):
        return Agent(
            role="Gathers and organizes information from various sources to provide comprehensive data outputs.",
            backstory=dedent("""The Information Manager is adept at scanning vast data sets to find relevant facts and figures."""),
            goal=dedent("""To efficiently collect and organize information, providing a reliable foundation for further analysis."""),
            allow_delegation=False,
            verbose=True,
            llm=self.Ollama,
            tools=[search_tool]
        )

    def evaluation_agent(self):
        return Agent(
            role="Analyzes gathered data, refines it, and presents it in a clear, user-friendly format with key insights.",
            backstory=dedent("""The Evaluation Agent specializes in transforming raw data into actionable insights and concise summaries."""),
            goal=dedent("""To present information in an accessible and aesthetically pleasing manner, maximizing user understanding and engagement."""),
            allow_delegation=False,
            verbose=True,
            llm=self.Ollama,
            tools=[search_tool]
        )
