from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain import OpenAI
 
db = SQLDatabase.from_databricks(catalog="samples", schema="nyctaxi")
llm = OpenAI(temperature=.7)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)

agent.run("What is the longest trip distance and how long did it take?")