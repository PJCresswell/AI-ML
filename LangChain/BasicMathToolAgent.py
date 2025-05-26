from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain.tools import StructuredTool
from pydantic import BaseModel

MODEL = 'gpt-4o-mini'

llm = ChatOpenAI(
    model=MODEL,
    temperature=0.2,
    n=1
)

# Define a Pydantic schema for the input arguments
class PythonReplInput(BaseModel):
    input: str

# Define the tool using StructuredTool with an args_schema
def run_python_code(input: str, **kwargs):
    try:
        # Safely evaluate the input string as a Python expression
        result = eval(input)
        return str(result)
    except Exception as e:
        return str(e)

repl_tool = StructuredTool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=run_python_code,  # Use the custom Python REPL function
    args_schema=PythonReplInput  # Define the expected input schema
)

prompt = hub.pull("hwchase17/openai-functions-agent")

tools = [repl_tool]
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Pass a string Python command as input
result = agent_executor.invoke({"input": "What is 8273 * 1821?"})

print(result)