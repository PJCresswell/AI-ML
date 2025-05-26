from langchain_openai import ChatOpenAI

MODEL = 'gpt-4o-mini'
llm = ChatOpenAI(
        model=MODEL,
        temperature=0.2,
        n=1
    )

# See math done badly by a LLM. Is doing it in it's head so to speak. It's wrong !
print(llm.invoke("What is 8273 times 1821?"))
print(8273 * 1821)

# Use the PythonREPL tool - is an execution shell
from langchain_experimental.utilities import PythonREPL

# Me calling the Python execution shell
python_repl = PythonREPL()
result = python_repl.run("print(1+1)")
print(result)

from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
# A structured tool is a runnable that can operate with any number of inputs
from langchain.tools import StructuredTool
# Pydantic is a library to check Python syntax
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

# Safely evaluate the input string as a Python expression
def run_python_code(input: str, **kwargs):
    try:
        result = eval(input)
        return str(result)
    except Exception as e:
        return str(e)

# Now using the tool - allows you to run Python
# Use the description to explain what the tool is to the LLM
# Define the tool using StructuredTool with an args_schema : can operate on any number of inputs
repl_tool = StructuredTool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=run_python_code,  # Use the custom Python REPL function
    args_schema=PythonReplInput  # Define the expected input schema
)

# Uses the same functions calling agent
prompt = hub.pull("hwchase17/openai-functions-agent")

tools = [repl_tool]
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Pass a string Python command as input
result = agent_executor.invoke({"input": "What is 8273 * 1821?"})

print(result)