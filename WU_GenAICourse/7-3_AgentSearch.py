#engine = 'duck'
engine = 'tavily'

from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor

MODEL = 'gpt-4o-mini'

llm = ChatOpenAI(
    model=MODEL,
    temperature=0.2,
    n=1
)

if engine == 'duck':
    from langchain_community.tools import DuckDuckGoSearchRun

    search_tool = DuckDuckGoSearchRun()
    result = search_tool.run("Who is the current president of the US ?")
    print(result)

    search_tool = DuckDuckGoSearchRun()

    # The same prompt template
    prompt = hub.pull("hwchase17/openai-functions-agent")

    tools = [search_tool]
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    agent_executor.invoke({"input": "Return the value of the DJIA as a floating point number, return just the number, no text or comments."})

if engine == 'tavily':
    from langchain_community.tools.tavily_search import TavilySearchResults

    # Just an example of using the search tool
    # Is pleasingly verbose
    tool = TavilySearchResults()
    result = tool.invoke({"query": "What happened in the latest burning man floods"})
    print(result)

    from langchain import hub
    from langchain.agents import create_tool_calling_agent
    from langchain.agents import AgentExecutor

    search_tool = TavilySearchResults()

    # The same prompt template
    prompt = hub.pull("hwchase17/openai-functions-agent")

    tools = [search_tool]
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    agent_executor.invoke({"input": "Who is the oldest world leader?"})