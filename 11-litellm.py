from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, SystemMessage
import os

litellm_key =os.getenv("LITELLM_VIRTUAL_KEY") 

chat = ChatOpenAI(
    base_url="https://litellm.cosmos.cboxlab.com",
    model = 'deepseek-v4-flash',
    temperature=0.1,
    api_key=litellm_key,
)

messages = [
    SystemMessage(
        content="You are a helpful assistant that im using to make a test request to."
    ),
    HumanMessage(
        content="test from litellm. tell me why it's amazing in 1 sentence"
    ),
]
response = chat.invoke(messages)

print(response)
