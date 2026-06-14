import requests

from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver


@dataclass
class Context:
    user_id: str

@dataclass
class ResponseFormat:
    summary: str
    temperature_celsius: float
    temperateure_fahrenheit: float
    humidity: float


@tool('get_current_weather', description='Get the current weather in a given city', return_direct=False)
def get_current_weather(city: str) -> str:
    """Get the current weather in a given city."""
    response = requests.get(f'https://wttr.in/{city}?format=j1')
    return response.json()

@tool('locate_user', description="Look up user's city based on the context")
def locate_user(runtime: ToolRuntime[Context]) -> str:
    """Look up user's city based on the context."""
    context: Context = runtime.context
    if context.user_id == 'user_1':
        return 'Geneve, Switzerland'
    else:
        return 'Unknown'

model = init_chat_model(
    model='deepseek-chat',
    temperature=0.7,
)

checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[get_current_weather, locate_user],
    system_prompt="You are a helpful assistant that can provide current weather information for any city, and also cracks jokes often",
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer
)

config = {
        'configurable': {
            'thread_id': "1"
            }
        }

agent.invoke({
    'messages': [
        {
            'role': 'user',
            'content': 'What is the current weather?'
            }
        ]
    },
    config = config,
    context = Context(user_id='user_1')
)


res = agent.invoke({
    'messages': [
        {
            'role': 'user',
            'content': 'And is this usual?'
            }
        ]
    },
    config = config,
    context = Context(user_id='user_1')
)

print(res['messages'][-1].content)
