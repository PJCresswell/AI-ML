from langchain.tools import StructuredTool
from pydantic import BaseModel
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor

# Define a Pydantic schema for the button press input
class ButtonPressInput(BaseModel):
    button_color: str

class CarTool:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.direction = 'north'
        self.directions = ['north', 'east', 'south', 'west']
        self.tool = StructuredTool(
            name="control_panel",
            description="""Three buttons, red, green and yellow that you can push.
            The red button moves you forward 1 unit.
            The green button turns left 90 degrees,
            the yellow button turns right 90 degrees.""",
            func=self.press_button,
            args_schema=ButtonPressInput  # Define the expected input schema
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

    def press_button(self, button_color: str):
        button_color = button_color.strip().lower()
        if button_color == 'red':
            self.move_forward()
            result = "The button glows red, you move forward one unit."
        elif button_color == 'green':
            self.turn_left()
            result = "The button glows green, you turn 90 degrees to the left."
        elif button_color == 'yellow':
            self.turn_right()
            result = "The button glows yellow, you turn 90 degrees to the right."
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

# Initialize LLM and create an agent
MODEL = 'gpt-4o-mini'
llm = ChatOpenAI(model=MODEL, temperature=0.2, n=1)

prompt = hub.pull("hwchase17/openai-functions-agent")

tools = [car_tool.tool]
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Invoke the agent to control the car
agent_executor.invoke({"input": "Push the buttons in a way that causes the car to move in a rectangle."})