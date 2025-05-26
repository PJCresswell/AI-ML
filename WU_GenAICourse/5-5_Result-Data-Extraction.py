# type = 'structured'
# type = 'csv'
# type = 'json'
# type = 'pandas'
# type = 'datetime'
# type = 'pydantic'
type = 'custom'

if type == 'structured':
    from langchain.output_parsers import ResponseSchema, StructuredOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI

    response_schemas = [
        ResponseSchema(name="answer", description="answer to the user's question"),
        ResponseSchema(
            name="source",
            description="source used to answer the user's question, should be a website.",
    )   ,
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    prompt = PromptTemplate(
        template="answer the users question as best as possible.\n{format_instructions}\n{question}",
        input_variables=["question"],
        partial_variables={"format_instructions": format_instructions},
    )
    question = "When was the Python programming language introduced?"
    print(prompt.invoke(question).text)

    MODEL = "gpt-4o-mini"
    TEMPERATURE = 0

    # Initialize the OpenAI LLM with your API key
    llm = ChatOpenAI(
        model=MODEL,
        temperature=TEMPERATURE,
        n=1
    )
    chain = prompt | llm | output_parser
    result = chain.invoke({"question": question})
    print(result)

if type == 'csv':
    from langchain.output_parsers import CommaSeparatedListOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI

    output_parser = CommaSeparatedListOutputParser()
    format_instructions = output_parser.get_format_instructions()
    prompt = PromptTemplate(
        template="List ten {subject}.\n{format_instructions}",
        input_variables=["subject"],
        partial_variables={"format_instructions": format_instructions},
    )
    MODEL = "gpt-4o-mini"
    TEMPERATURE = 0
    # Initialize the OpenAI LLM with your API key
    llm = ChatOpenAI(
        model=MODEL,
        temperature=TEMPERATURE,
        n=1
    )
    chain = prompt | llm | output_parser
    result = chain.invoke({"subject": "cities"})
    print(result)

if type == 'json':
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_core.pydantic_v1 import BaseModel, Field
    from langchain_openai import ChatOpenAI

    # Define your desired data structure.
    class Translate(BaseModel):
        detected: str = Field(description="the detected language of the input")
        spanish: str = Field(description="the input translated to Spanish")
        french: str = Field(description="the input translated to French")
        chinese: str = Field(description="the input translated to Chinese")

    # And a query intented to prompt a language model to populate the data structure.
    input_text = "What is your name?"

    # Set up a parser + inject instructions into the prompt template.
    parser = JsonOutputParser(pydantic_object=Translate)
    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{input}\n",
        input_variables=["input"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    MODEL = "gpt-4o-mini"
    TEMPERATURE = 0
    # Initialize the OpenAI LLM with your API key
    llm = ChatOpenAI(
        model=MODEL,
        temperature=TEMPERATURE,
        n=1
    )
    chain = prompt | llm | parser
    result = chain.invoke({"input": input_text})
    print(result)

if type == 'pandas':
    import pprint
    from typing import Any, Dict

    import pandas as pd
    from langchain.output_parsers import PandasDataFrameOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI

    # Load the iris dataset
    df = pd.read_csv("https://data.heatonresearch.com/data/t81-558/iris.csv", na_values=["NA", "?"])
    print(df.head())

    parser = PandasDataFrameOutputParser(dataframe=df)
    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    MODEL = "gpt-4o-mini"
    TEMPERATURE = 0
    # Initialize the OpenAI LLM with your API key
    llm = ChatOpenAI(
        model=MODEL,
        temperature=TEMPERATURE,
        n=1
    )
    chain = prompt | llm | parser
    query = "Get the mean of the sepal_l column."
    parser_output = chain.invoke({"query": query})
    print(parser_output)
    query = "Get the sum of the petal_w column."
    parser_output = chain.invoke({"query": query})
    print(parser_output)

if type == 'datetime':
    from langchain.output_parsers import DatetimeOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI

    output_parser = DatetimeOutputParser()
    template = """Answer the users question:

    {question}

    {format_instructions}"""
    prompt = PromptTemplate.from_template(
        template,
        partial_variables={"format_instructions": output_parser.get_format_instructions()},
    )
    print(prompt)
    MODEL = "gpt-4o-mini"
    TEMPERATURE = 0
    # Initialize the OpenAI LLM with your API key
    llm = ChatOpenAI(
        model=MODEL,
        temperature=TEMPERATURE,
        n=1
    )
    chain = prompt | llm | output_parser
    output = chain.invoke({"question": "What is the date of the war in the video game Fallout?"})
    print(output)

if type == 'pydantic':
    from typing import List

    from langchain.output_parsers import PydanticOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_core.pydantic_v1 import BaseModel, Field, validator
    from langchain_openai import ChatOpenAI

    MODEL = "gpt-4o-mini"
    TEMPERATURE = 0
    # Initialize the OpenAI LLM with your API key
    llm = ChatOpenAI(
        model=MODEL,
        temperature=TEMPERATURE,
        n=1
    )

    class Actor(BaseModel):
        name: str = Field(description="name of an actor")
        film_names: List[str] = Field(description="list of names of films they starred in")

        @validator('name')
        def validate_name(cls, value):
            parts = value.split()
            if len(parts) < 2:
                raise ValueError("Name must contain at least two words.")
            if not all(part[0].isupper() for part in parts):
                raise ValueError("Each word in the name must start with a capital letter.")
            return value

    actor_query = "Generate the filmography for a random actor."
    parser = PydanticOutputParser(pydantic_object=Actor)
    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser
    result = chain.invoke({"query": actor_query})
    print(result)

if type == 'custom':
    from langchain_openai import ChatOpenAI

    MODEL = 'gpt-4o-mini'
    TEMPERATURE = 0.0

    # Initialize the OpenAI LLM with your API key
    llm = ChatOpenAI(
        model=MODEL,
        temperature=TEMPERATURE,
        n=1
    )
    from typing import Iterable
    from langchain_core.messages import AIMessage, AIMessageChunk

    def parse(ai_message: AIMessage) -> str:
        """Parse the AI message."""
        return ai_message.content.swapcase()

    chain = llm | parse
    result = chain.invoke("hello")
    print(result)

    import re

    def extract_python_code(mixed_text):
        code_blocks = re.findall(r'```python(.*?)```', mixed_text, re.DOTALL)
        return "\n".join(code_blocks).strip()


    mixed_text = """
    Yes, you can estimate the value of Pi using various methods in Python. One
    common approach is the Monte Carlo method. Here's a simple example:

    ```python
    import random

    def estimate_pi(num_samples):
        inside_circle = 0

        for _ in range(num_samples):
            x = random.uniform(0, 1)
            y = random.uniform(0, 1)
            distance = x**2 + y**2

            if distance <= 1:
                inside_circle += 1

        pi_estimate = (inside_circle / num_samples) * 4
        return pi_estimate

    num_samples = 1000000
    pi_estimate = estimate_pi(num_samples)
    print(f"Estimated value of Pi: {pi_estimate}")
    ```

    This code uses the Monte Carlo method to estimate Pi by generating random points
    within a unit square and checking how many fall inside a quarter circle. The
    ratio of points inside the circle to the total points, multiplied by 4, gives an
    estimate of Pi.

    Would you like to explore other methods or need further explanation on this
    approach?

    """

    python_code = extract_python_code(mixed_text)
    print(python_code)

    from langchain_core.output_parsers import BaseOutputParser

    class CodeOutputParser(BaseOutputParser[str]):
        """Custom code parser."""

        def parse(self, text):
            return extract_python_code(text)

        @property
        def _type(self) -> str:
            return "CodeOutputParser"

    parser = CodeOutputParser()
    chain = llm | parser
    result = chain.invoke("Can I create Python code to estimate the value of Pi.")
    print(result)