from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import TextLoader
from langchain import OpenAI, PromptTemplate
from langchain_openai import ChatOpenAI
from IPython.display import display_markdown
from langchain.indexes import VectorstoreIndexCreator
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.inmemory import InMemoryVectorStore
from langchain.schema import Document
import requests

# Location of the information to be used
urls = [
    "https://data.heatonresearch.com/data/t81-559/bios/DD.txt",
    "https://data.heatonresearch.com/data/t81-559/bios/FT.txt",
    "https://data.heatonresearch.com/data/t81-559/bios/GS.txt",
    "https://data.heatonresearch.com/data/t81-559/bios/NGS.txt",
    "https://data.heatonresearch.com/data/t81-559/bios/TI.txt"
]

def chunk_text(text, chunk_size, overlap):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

chunk_size = 900
overlap = 50

documents = []

# For each document, gets the text and breaks up into individual document objects of 900 chars with an overlap
# Concatenates all of these document objects together
for url in urls:
    print(f"Reading: {url}")
    response = requests.get(url)
    response.raise_for_status()  # Ensure we notice bad responses
    content = response.text
    chunks = chunk_text(content, chunk_size, overlap)
    for chunk in chunks:
        document = Document(page_content=chunk)
        documents.append(document)
# print(documents)

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain.vectorstores import Chroma

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(docs, embedding_function)
retriever = db.as_retriever()

#result = retriever.invoke("who is Samantha Doyle")[0]
#print('*** Starting Chroma results***')
#print(result)
#print('*** Ending Chroma results***')

from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Search for information about people who work for several companies. For any questions about people you do not know, you must use this tool!",
)

from langchain import hub
from langchain.agents import create_tool_calling_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import AgentExecutor

MODEL = 'gpt-4o-mini'

llm = ChatOpenAI(
        model=MODEL,
        temperature=0.2,
        n=1
    )

search_tool = DuckDuckGoSearchRun()

prompt = hub.pull("hwchase17/openai-functions-agent")

tools = [search_tool, retriever_tool]
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# Still a single request but works out needs to use the retrieve tool to find out who Samantha Doyle is, then search the web for her career prospects
result = agent_executor.invoke({"input": "What are the job prospects in 2024 for Samantha Doyle's career? Return just a 2-sentence assessment of her career future."})
print('*** Starting Search results***')
print(result)
print('*** Ending Search results***')