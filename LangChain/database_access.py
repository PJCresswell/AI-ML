import pandas as pd
import sqlite3

'''
# Create a database and populate with the portfolio information
df = pd.read_csv('data\Portfolio.csv')
print(df.head())
connection = sqlite3.connect("portfolio.db")
df.to_sql(name='portfolio', con=connection)
connection.commit()
connection.close()

# Example where we load the data back in
con = sqlite3.connect("portfolio.db")
cur = con.cursor()
res = cur.execute("SELECT * FROM portfolio")
result = res.fetchall()
print(result)
con.close()
'''

from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI

db = SQLDatabase.from_uri("sqlite:///portfolio.db")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=1)

agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

agent_executor.invoke("Which stock contributes the most to the overall portfolio value for desk 1?")
