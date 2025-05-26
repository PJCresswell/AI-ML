#######################
# Test the environment
#######################

import platform
import sys
import pandas as pd
import sklearn as sk
import torch
has_gpu = torch.cuda.is_available()
has_mps = torch.backends.mps.is_built()
device = "mps" if has_mps else "cuda" if has_gpu else "cpu"
print(f"Python Platform: {platform.platform()}")
print(f"PyTorch Version: {torch.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
print("GPU is", "available" if has_gpu else "NOT AVAILABLE")
print("MPS (Apple Metal) is", "AVAILABLE" if has_mps else "NOT AVAILABLE")
print(f"Target device is {device}")

'''
from langchain_openai import OpenAI
# Initialize the OpenAI LLM with your API key
llm = OpenAI()
# Ask the language model a question, ensuring the prompt is in a list
question = ["Are you working properly?"]
response = llm.invoke(question)
print(response)
'''

#######################
# First Langchain project
#######################

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

# Create the two prompt templates
title_template = PromptTemplate( input_variables = ['topic'], template = 'Give me a blog post title on {topic} in English' )
article_template = PromptTemplate( input_variables = ['title'], template = 'Write a blog post for {title}' )

MODEL = 'gpt-4o-mini'
# Create a chain to generate a random
llm = ChatOpenAI(model=MODEL, temperature=0.7)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True)

MODEL2 = 'gpt-4'
# Create the article chain
llm2 = ChatOpenAI(model=MODEL2, temperature=0.1)
article_chain = LLMChain(llm=llm2, prompt=article_template, verbose=True)

# Create a complete chain to create a new blog post
complete_chain=SimpleSequentialChain(chains=[title_chain, article_chain], verbose=True)
article = complete_chain.invoke('motorbikes')
print(article)