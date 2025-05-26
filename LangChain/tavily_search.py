
from langchain.agents import create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain.agents import AgentExecutor

llm = ChatOpenAI(model="gpt-4o-mini", temperature=1)

search_tool = TavilySearchResults(max_results=3)
tools = [search_tool]

from langchain import hub
prompt = hub.pull("hwchase17/openai-functions-agent")

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": "What is the latest stock price for BAC?"})

