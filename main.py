from typing import List

from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# from tavily import TavilyClient
from langchain_tavily import TavilySearch

load_dotenv()


class Source(BaseModel):
    """Schema for a source used by the agent """

    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for the agent response with answer and sources"""

    answer:str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory="list of sources to generate the answer")

# tavily = TavilyClient()

# @tool
# def search(query: str) -> str:
#    """
#    Tool that searches over the internet
#    Args:
#        query: The query to search for
#    Returns:
#        The search result
#    """
#    print(f"Searching for {query}")
#    #return "Tokyo weather is sunny"
#    return tavily.search(query=query)

# tools=[search]
tools = [TavilySearch()]

# llm = ChatOllama(temperature=0, model="gemma3:270m")
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
#llm = ChatOpenAI(model="gpt-5", temperature=0)
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-course!")
    # response = agent.invoke({"messages": [("user", "What is the weather in Tokyo?")]})
    # result = agent.invoke({"messages": HumanMessage(content="What is the weather in Tokyo?")})
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="search for 3 job postings for an ai engineer using langchain in the bay are on linkedin and list their details"
            )
        }
    )
    print(result)


if __name__ == "__main__":
    main()
