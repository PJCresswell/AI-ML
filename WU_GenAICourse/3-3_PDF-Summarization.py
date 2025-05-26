from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain import OpenAI, PromptTemplate
from langchain_openai import ChatOpenAI

MODEL = 'gpt-4o-mini'

llm = ChatOpenAI(
    model=MODEL,
    temperature=0.2,
    n=1
)

chain = load_summarize_chain(llm, chain_type="map_reduce")

url = "https://arxiv.org/pdf/1706.03762"
loader = PyPDFLoader(url)
docs = loader.load_and_split()
summary = chain.invoke(docs)['output_text']
print(summary)