from re import VERBOSE
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain.agents import AgentExecutor

MODEL = 'gpt-4o-mini'

llm = ChatOpenAI(
        model=MODEL,
        temperature=0.2,
        n=1
    )

search_tool = TavilySearchResults()
prompt = hub.pull("hwchase17/openai-functions-agent")
tools = [search_tool]
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor.invoke({"input": "Who is the president of the US as of 24 January 2025?"})

#search_tool = DuckDuckGoSearchRun()
## Simple prompt template
#prompt = hub.pull("hwchase17/openai-functions-agent")
#tools = [search_tool]
#agent = create_tool_calling_agent(llm, tools, prompt)
#agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
#agent_executor.invoke({"input": "What is the currnet value of the DJIA?"})
