#########################
# Simple example of a tool calling agent. Single step - no chaining
#########################

from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import AgentExecutor

# Set up the LLM model
MODEL = 'gpt-4o-mini'
llm = ChatOpenAI(
    model=MODEL,
    temperature=0.2,
    n=1
)

# Define the search tool that the agent will use
search_tool = DuckDuckGoSearchRun()

# Can create your own prompts. This is a simple one that just says you are a helpful assistant
# https://smith.langchain.com/hub/hwchase17/openai-functions-agent

prompt = hub.pull("hwchase17/openai-functions-agent")

tools = [search_tool]
# Create the tool calling agent
agent = create_tool_calling_agent(llm, tools, prompt)
# Create the runtime that will call the agent, run the actions, pass the results back etc
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# The actual query. Again, simple call and response - no chains
result = agent_executor.invoke({"input": "What is the value of the S&P500 as of 30 Sep 2024 ?"})
print(result)