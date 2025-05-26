import requests

def get_email(i):
    if i<1 or i>10:
        raise Exception("Invalid email number")

    # URL to download
    url = f"https://data.heatonresearch.com/wustl/CABI/genai-langchain/emails/email_{i}.txt"

    # Perform a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Convert the content of the response to a string
        content = response.text
        return content
    else:
        raise Exception("Failed to retrieve the content")

print(get_email(1))

from langchain_openai import ChatOpenAI

MODEL = 'gpt-4o-mini'
TEMPERATURE = 0.0

# Initialize the OpenAI LLM with your API key
llm = ChatOpenAI(
    model=MODEL,
    temperature=TEMPERATURE,
    n=1
)

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

email_prompt = PromptTemplate( input_variables = ['email'], template = """
Classify the following email as either:
* spam - For marketing emails trying to sell something
* faculty - For faculty annoucements and requests
* help - For students requesting help on an assignment
* lor - For students requesting a letter of recommendation
* other - If it does not fit into any of these.
Here is the email. Return code, such as spam. Return nothing else, do not explain your choice.
Make sure that if the email does not fit into one of the categories that you classify it as other.
Here is the email:

{email}""")

help_prompt = PromptTemplate( input_variables = ['email'], template = """
You are given an email where a student is asking about an assignment. Return the assignment number
that they are asking about. If you cannot tell return a ?. Return only the assignment number as
an integer, do not explain.
Here is the email:

{email}""")

chain_email = email_prompt | llm
chain_help = help_prompt | llm

for i in range(1,11):
    email = get_email(i)
    classification = chain_email.invoke(email).content.strip()
    if classification == 'help':
        assignment = chain_help.invoke(email).content.strip()
        print(f"Email #{i} is a question about assignment #{assignment}")
    else:
        print(f"Email #{i} is: {classification}")