
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage

model = init_chat_model(
    model='deepseek-v4-flash',
    temperature=0.7,
)

conversation = [
    SystemMessage(content="You are a helpful assistant for questions reqarding programming"),
    HumanMessage(content="What is the difference between a list and a tuple in Python?"),
    AIMessage(content="A list is a mutable data structure in Python, which means that you can change its contents after it has been created. A tuple, on the other hand, is an immutable data structure, which means that once it has been created, you cannot change its contents. Lists are defined using square brackets [], while tuples are defined using parentheses ()."),
    HumanMessage(content="Can you give me an example of a list and a tuple?"),
]

# Full response in one shot
#  response = model.invoke(conversation)

# Streaming response
for chunk in model.stream(conversation):
    print(chunk.text, end='', flush=True)
