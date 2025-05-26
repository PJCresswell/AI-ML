from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import PromptTemplate
from IPython.display import display_markdown
import pickle

'''
# Conversation memory buffer - stores the last 5 interactions

from langchain.memory import ConversationBufferWindowMemory
memory = ConversationBufferWindowMemory(k=5)
    
# Conversation token buffer - stores the last 2048 tokens
from langchain.memory import ConversationTokenBufferMemory
memory = ConversationTokenBufferMemory(llm=summary_llm,max_token_limit=2048)

# Conversation summary memory - uses another LLM to summarise the conversation
from langchain.memory import ConversationSummaryMemory
memory = ConversationSummaryMemory(llm=summary_llm)

# For all 
conversation = ConversationChain(
    prompt=PROMPT_TEMPLATE,
    llm=llm,
    memory=memory,
    verbose=False
)
def converse(conversation, prompt):
    output = conversation.invoke(prompt)
    return output['response']
'''


DEFAULT_TEMPLATE = """You are a helpful assistant. Format answers with markdown.

Current conversation:
{history}
Human: {input}
AI:"""

MODEL = 'gpt-4o-mini'

class ChatBot:
    def __init__(self, llm_chat, llm_summary, template):
        """
        Initializes the ChatBot with language models and a template for conversation.

        :param llm_chat: A large language model for handling chat responses.
        :param llm_summary: A large language model for summarizing conversations.
        :param template: A string template defining the conversation structure.
        """
        self.llm_chat = llm_chat
        self.llm_summary = llm_summary
        self.template = template
        self.prompt_template = PromptTemplate(input_variables=["history", "input"], template=self.template)

        # Initialize memory and conversation chain
        self.memory = ConversationSummaryMemory(llm=self.llm_summary)
        self.conversation = ConversationChain(
            prompt=self.prompt_template,
            llm=self.llm_chat,
            memory=self.memory,
            verbose=False
        )

        self.history = []

    def converse(self, prompt):
        """
        Processes a conversation prompt and updates the internal history and memory.

        :param prompt: The input prompt from the user.
        :return: The generated response from the language model.
        """
        self.history.append([self.memory.buffer, prompt])
        output = self.conversation.invoke(prompt)
        return output['response']

    def chat(self, prompt):
        """
        Handles the full cycle of receiving a prompt, processing it, and displaying the result.

        :param prompt: The input prompt from the user.
        """
        print(f"Human: {prompt}")
        output = self.converse(prompt)
        display_markdown(output, raw=True)

    def print_memory(self):
        """
        Displays the current state of the conversation memory.
        """
        print("**Memory:")
        print(self.memory.buffer)

    def clear_memory(self):
        """
        Clears the conversation memory.
        """
        self.memory.clear()

    def undo(self):
        """
        Reverts the conversation memory to the state before the last interaction.
        """
        if len(self.history) > 0:
            self.memory.buffer = self.history.pop()[0]
        else:
            print("Nothing to undo.")

    def regenerate(self):
        """
        Re-executes the last undone interaction, effectively redoing an undo operation.
        """
        if len(self.history) > 0:
            self.memory.buffer, prompt = self.history.pop()
            self.chat(prompt)
        else:
            print("Nothing to regenerate.")

    def save_history(self, file_path):
        """
        Saves the conversation history to a file using pickle.

        :param file_path: The file path where the history should be saved.
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self.history, f)

    def load_history(self, file_path):
        """
        Loads the conversation history from a file using pickle.

        :param file_path: The file path from which to load the history.
        """
        with open(file_path, 'rb') as f:
            self.history = pickle.load(f)
            # Optionally reset the memory based on the last saved state
            if self.history:
                self.memory.buffer = self.history[-1][0]

# Initialize the OpenAI LLM with your API key
llm = ChatOpenAI(
  model=MODEL,
  temperature= 0.3,
  n= 1)

#c = ChatBot(llm, llm, DEFAULT_TEMPLATE)
#c.chat("Hello, my name is Jeff.")
#c.chat("I like coffee.")
#c.chat("What is my name.")
#c.chat("Do I like coffee?")
#c.save_history('history.pkl')

c = ChatBot(llm, llm, DEFAULT_TEMPLATE)
c.load_history('history.pkl')
c.chat("What is my name?")

print(c.memory.buffer)