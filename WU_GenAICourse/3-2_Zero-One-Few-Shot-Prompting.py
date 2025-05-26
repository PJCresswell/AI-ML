from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI

MODEL = 'gpt-4o-mini'
TEMPERATURE = 0.2

def get_response(llm, prompt):
  messages = [
      SystemMessage(
          content="You are a helpful assistant that answers questions accurately."
      ),
      HumanMessage(content=prompt),
  ]
  print("Model response:")
  output = llm.invoke(messages)
  return output


# Initialize the OpenAI LLM with your API key
llm = ChatOpenAI(
  model=MODEL,
  temperature=TEMPERATURE,
  n= 1,
  max_tokens= 256)

'''
###############################
# Zero shot - no examples given
###############################

print(get_response(llm, """
Generate a positive letter of recommendation for John Smith, a student of mine
for INFO 558 at Washington University, my name is Jeff Heaton. He is applying
for a Master of Science in Computer Science. Just give me the
body text of the letter, no header or footer. Format in markdown.
Below is his request.

I hope this message finds you well and that you are enjoying the holiday season!
I am John Smith (ID: 1234), a proud alumnus of WashU, having graduated in
January 2021 with a Master’s degree in Quantitative Finance.

During the spring semester of 2020, I had the pleasure of attending your course,
INFO 558: Applications of Deep Neural Networks, which was an elective for my
master's program. I thoroughly enjoyed the content and was deeply engaged
throughout, culminating in an A+ grade.

Since graduating with a 3.99 GPA—top of my major—I have been working as a Senior
Financial Risk Analyst at RGA. My role primarily involves developing automation
tools and programming for strategic analysis and other analytical tasks. To
further enhance my programming skills and knowledge, I am planning to pursue a
part-time Master's in Computer Science while continuing to work at RGA.

I am a great admirer of your work (I’m a regular viewer of your YouTube channel
and have recommended it to my colleagues), and your insights would be invaluable
in my application. I am applying to the following programs:

Georgia Tech, Master of Science in Computer Science
University of Pennsylvania, Master of Computer & Information Technology
Could I possibly ask for your support with a recommendation letter for these
applications? I have attached my resume for your reference and am happy to
provide any additional information you might need.

Thank you very much for considering my request. I look forward to your
positive response.

Warm regards,

John
"""))

###############################
# Few shot - a set of examples given
###############################

print(get_response(llm, """
Generate a positive letter of reccomendation for John Smith, a student of mine
for INFO 558 at Washington University, my name is Jeff Heaton. He is applying
for a Master of Science in Computer Science. Just give me the
body text of the letter, no header or footer. Format in markdown.

-----------------
Examples of letters of reccomendation, written by me.

To Whom It May Concern:
John earned an A+ in my course Applications of Deep Neural Networks for the
Fall 2019 semester at Washington University in St. Louis. During the semester
I got a chance to know John through several discussions, both about my course
and his research interests. While John did not come from a computer science
background he has demonstrated himself as a capable Python programmer and was
able to express his ideas in code.  My primary career is as a VP of data science
at RGA, a Fortune 500 insurance company.  In this role I know the value of
individuals, such as John, who have a background in finance, understand
advanced machine learning topics, and can code sufficiently well to function
as a data scientist.

John was a student that in my class, T81-558: Application of Deep Neural Networks,
for the Spring 2017 semester. This is a technical graduate class which includes
students from the Masters of Science lnformation Systems, Management,
computer science, and other disciplines. The course teaches students to
implement deep neural networks using Google TensorFlow and Keras in the Python
programming language. Students are expected to complete four computer programs
and complete a final project. John did well in my course and earned an A+ (4.0).

-----------
The details of this student's request follows.

I hope this message finds you well and that you are enjoying the holiday season!
I am John Smith (ID: 1234), a proud alumnus of WashU, having graduated in
January 2021 with a Master’s degree in Quantitative Finance.

During the spring semester of 2020, I had the pleasure of attending your course,
INFO 558: Applications of Deep Neural Networks, which was an elective for my
master's program. I thoroughly enjoyed the content and was deeply engaged
throughout, culminating in an A+ grade.

Since graduating with a 3.99 GPA—top of my major—I have been working as a Senior
Financial Risk Analyst at RGA. My role primarily involves developing automation
tools and programming for strategic analysis and other analytical tasks. To
further enhance my programming skills and knowledge, I am planning to pursue a
part-time Master's in Computer Science while continuing to work at RGA.

I am a great admirer of your work (I’m a regular viewer of your YouTube channel
and have recommended it to my colleagues), and your insights would be invaluable
in my application. I am applying to the following programs:

Georgia Tech, Master of Science in Computer Science
University of Pennsylvania, Master of Computer & Information Technology
Could I possibly ask for your support with a recommendation letter for these
applications? I have attached my resume for your reference and am happy to
provide any additional information you might need.

Thank you very much for considering my request. I look forward to your
positive response.

Warm regards,

John
"""))
'''

###############################
# Generating sample data
###############################

def get_response2(llm, prompt):
    messages = [
        SystemMessage(
            content="""
            You are a helpful assistant that generates synthetic data for a person in the career
            field you are given. Provide a short bio for the person, not longer than
            5 sentences. No markdown. Do not mention the job title specifically."""
        ),
        HumanMessage(content=prompt),
    ]
    response = llm.invoke(messages)
    return response.content

CAREER = [
    "software engineer",
    "pediatric nurse",
    "financial analyst",
    "high school science teacher",
    "marketing manager"
]

import csv
import random
from tqdm import tqdm  # Progress bar library

FILENAME = "jobs.csv"

# Writing to the CSV file
with open(FILENAME, 'w', newline='\n') as csvfile:
    csvwriter = csv.writer(csvfile)

    # Use tqdm to show progress bar
    for i in tqdm(range(5), desc="Generating Careers"):
      career_choice = random.choice(CAREER)  # Randomly select a career
      csvwriter.writerow([i+1, get_response2(llm, career_choice)])