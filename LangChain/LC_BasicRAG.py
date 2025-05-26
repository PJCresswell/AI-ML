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

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import (SentenceTransformerEmbeddings)
from langchain.vectorstores import Chroma

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

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
overlap = 300

documents = []

for url in urls:
    print(f"Reading: {url}")
    response = requests.get(url)
    response.raise_for_status()  # Ensure we notice bad responses
    content = response.text
    chunks = chunk_text(content, chunk_size, overlap)
    for chunk in chunks:
        document = Document(page_content=chunk)
        documents.append(document)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
print(docs)

exit()

db = Chroma.from_documents(docs, embedding_function, persist_directory="/content/chroma_db")

print(type(db))

db2 = Chroma(persist_directory="/content/chroma_db", embedding_function=embedding_function)

from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

MODEL = 'gpt-4o-mini'

llm = ChatOpenAI(
        model=MODEL,
        temperature=0.2,
        n=1
    )

rag_prompt = hub.pull("rlm/rag-prompt")

def format_documents(documents):
    return "\n\n".join(doc.page_content for doc in documents)

retriever = db2.as_retriever()

qa_chain = (
    {"context": retriever | format_documents, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

result = qa_chain.invoke("What company does Elena Martinez work for?")
print(result)