from langchain_openai import ChatOpenAI

MODEL = 'gpt-4o-mini'

llm = ChatOpenAI(
    model=MODEL,
    temperature=0.2,
    n=1
)

from langchain.agents import Tool

def press_button(value):
  print(f"*************{value}")
  return "The button glows red!"

button_tool = Tool(
    name="button_machine",
    description="A big red button that you can push. You can send a single value to the button, if asked. ",
    func=press_button,
)

from langchain import hub
from langchain.agents import create_tool_calling_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import AgentExecutor

prompt = hub.pull("hwchase17/openai-functions-agent")

tools = [button_tool]
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor.invoke({"input": "Push the big red button, tell me what happens."})

from langchain.agents import Tool

class CarTool:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.direction = 'north'
        self.directions = ['north', 'east', 'south', 'west']
        self.tool = Tool(
            name="control_panel",
            description="""Three buttons, red, green and yellow that you can push.
            The red button moves you forward 1 unit.
            The green button turns left 90 degrees,
            the yellow button turns right 90 degrees.""",
            func=self.press_button,
        )

    def move_forward(self):
        if self.direction == 'north':
            self.y += 1
        elif self.direction == 'east':
            self.x += 1
        elif self.direction == 'south':
            self.y -= 1
        elif self.direction == 'west':
            self.x -= 1

    def turn_left(self):
        current_index = self.directions.index(self.direction)
        self.direction = self.directions[(current_index - 1) % 4]

    def turn_right(self):
        current_index = self.directions.index(self.direction)
        self.direction = self.directions[(current_index + 1) % 4]

    def press_button(self, button_color):
        button_color = button_color.strip().lower()
        if button_color == 'red':
            self.move_forward()
            result = "The button glows red, you move forward one unit."
        elif button_color == 'green':
            self.turn_left()
            result = "The button glows green, you turn 90 degrees to the right."
        elif button_color == 'yellow':
            self.turn_right()
            result = "The button glows yellow, you turn 90 degrees to the left."
        else:
            result = "The button buzzes, error."

        print(f"Current car position: {self.get_position()}")
        return result

    def get_position(self):
        return self.x, self.y, self.direction

# Example usage
car_tool = CarTool()

try:
    print(car_tool.tool.func('red'))  # Move forward
    print(car_tool.tool.func('green'))  # Turn left
    print(car_tool.tool.func('yellow'))  # Turn right
    print(car_tool.tool.func('blue'))  # Invalid button color, raises ValueError
except ValueError as e:
    print(e)

from langchain import hub
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor

#MODEL = 'gpt-4-turbo'
MODEL = 'gpt-3.5-turbo'

prompt = hub.pull("hwchase17/openai-functions-agent")

car_tool = CarTool()

tools = [car_tool.tool]
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor.invoke({"input": "Push the buttons in a way that causes the car to move in a rectangle."})