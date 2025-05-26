# Load in the ISDA Master Agreement Document
from langchain_community.document_loaders import PyPDFLoader
pdf_loader = PyPDFLoader(file_path='data/ISDA_Master_Agreement.pdf')
doc1 = pdf_loader.load()

# Split the document into chunks that can be quickly retrieved and integrated into prompts
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
chunks = splitter.split_documents(doc1)

# Embed this information into a Chroma vector store
from langchain_openai import OpenAIEmbeddings
embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')
from langchain_chroma import Chroma

# vector_store = Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory="data/chroma_db")
vector_store = Chroma(persist_directory="data/chroma_db", embedding_function=embedding_model)

# To use this information for RAG, we need three components. First, our LLM
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.2)

# Next the retriever to take the user input and retrieve the relevant document chunks
retriever = vector_store.as_retriever(search_type = 'similarity', search_kwargs={'k': 2})

# Finally a prompt - to combine the user input and document chunks
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("""
Use the following context to answer the question at the end.
Context: {context}
Question: {question}
""")

# Now we can define our chain
# RunnablePassThrough : Passes through the input
# StrOutPutParser : Formats the response to be just the content

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

chain = (
    {'context': retriever, 'question': RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

result = chain.invoke('How does this ISDA Master agreement define a ratings event?')
print(result)