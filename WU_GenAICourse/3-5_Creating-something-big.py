from langchain.chains.summarize import load_summarize_chain
from langchain import OpenAI, PromptTemplate
from langchain_openai import ChatOpenAI
from IPython.display import display_markdown

MODEL = 'gpt-4o-mini'

llm = ChatOpenAI(
    model=MODEL,
    temperature=0.7,
    n=1
)

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI

def query_llm(prompt):
    messages = [
        SystemMessage(
            content="You are assistant helping to write a book."
        ),
        HumanMessage(
            content=prompt
        ),
    ]
    output = llm.invoke(messages)
    return output.content

SUBJECT = "international spy thriller"

title = query_llm(f"""
Give me a random title for a book on the subject '{SUBJECT}'.
Return only the title, no additional text.
""").strip(" '\"")
print(title)

synopsis = query_llm(f"""
Give me a synopsis for a book of the title '{SUBJECT}' for a book on the subject '{SUBJECT}'.
Return only the synopsis, no additional text.
""").strip(" '\"")
print(synopsis)

toc = query_llm(f"""
Give me a table of contents for a book of the title '{title}' for a book on
the subject '{SUBJECT}' the book synopsis is '{synopsis}'.
It is a short book of three chapters.
Return the table of contents as a list of chapter titles.
Separate the chapter number and chapter title with a pipe character '|'.
Return only the chapter names, no additional text.
""").strip(" '\"")
print(toc)

# Split the string into lines
lines = toc.splitlines()

# Extract titles using list comprehension
toc2 = [line.split('|')[1].strip() for line in lines if line]

# Print the list of titles
print(toc2)

def render_chapter(num, chapter_title, title, subject, synopsis, toc):
    txt = query_llm(f"""
    Write Chapter {num}, titled "{chapter_title}" for a book of the title '{title}' for a book on
    the subject '{subject}' the book synopsis is '{synopsis}' the table of contents is '{toc}'.
    Give me only the chapter text, no chapter heading, no chapter title, number, no additional text.
    """).strip(" '\"")
    return txt

txt = render_chapter(1, toc2[0], title, SUBJECT, synopsis, toc)
print(txt)

book = ""

# Render the title and synopsis
book += f"# {title}\n"
book += f"{synopsis}\n"

# Render the toc
book += f"\n## Table of Contents\n\n"
num = 1
for chapter_title in toc2:
    book += f"{num}. {chapter_title}\n"
    num += 1

# Render the book
chapter = 1
for chapter_title in toc2:
    print(f"Rendering chapter {chapter}/{len(toc2)}: {chapter_title}")
    txt = render_chapter(chapter, chapter_title, title, SUBJECT, synopsis, toc)
    book += f"\n\n## Chapter {chapter}: {chapter_title}\n"
    book += f"{txt}\n"
    chapter += 1

import markdown
import pdfkit

# Convert Markdown to HTML
html = markdown.markdown(book)

# Convert HTML to PDF
pdfkit.from_string(html, 'output.pdf')