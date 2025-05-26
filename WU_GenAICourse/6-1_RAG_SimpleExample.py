from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores.inmemory import InMemoryVectorStore
from langchain_community.llms import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator

MODEL = 'gpt-4o-mini'

llm = ChatOpenAI(
    model=MODEL,
    temperature=0.2,
    n=1
)

urls = [
  "https://arxiv.org/pdf/1706.03762",
  "https://arxiv.org/pdf/1810.04805",
  "https://arxiv.org/pdf/2005.14165",
  "https://arxiv.org/pdf/1910.10683"
]

loaders = []
chain = load_summarize_chain(llm, chain_type="map_reduce")
for url in urls:
    print(f"Reading: {url}")
    loader = PyPDFLoader(url)
    loaders.append(loader)
embeddings_model = OpenAIEmbeddings()
index = VectorstoreIndexCreator(embedding=embeddings_model,vectorstore_cls=InMemoryVectorStore).from_loaders(loaders)

query = "Which figure demonstrates Scaled Dot-Product Attention?"
result = index.query(query,llm=llm)
print(result)