from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.tools import create_retriever_tool
from langchain_core.vectorstores import InMemoryVectorStore
#  from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings


docs = [
    Document(
        page_content="""
Project Atlas Pineapple is an internal demo service used only in this RAG example.
If Atlas Pineapple reports error code AP-42, the runbook says to rotate the blue
service token, restart the worker named mango-runner, and then check dashboard panel 7.
The secret confirmation phrase for this fix is "silver kiwi".
""",
        metadata={"source": "demo-runbook", "service": "atlas-pineapple"},
    ),
    Document(
        page_content="""
Project Nebula Turnip is an internal demo service used only in this RAG example.
If Nebula Turnip reports error code NT-13, the runbook says to clear the queue
named turnip-priority, redeploy the scheduler named orbit-scheduler, and notify
the fictional on-call team Delta Lantern.
The secret confirmation phrase for this fix is "orange comet".
""",
        metadata={"source": "demo-runbook", "service": "nebula-turnip"},
    ),
    Document(
        page_content="""
Project Quartz Banana is an internal demo service used only in this RAG example.
If Quartz Banana reports error code QB-88, the runbook says to disable cache shard
banana-east-3, warm the replacement shard banana-west-1, and watch metric qb.latency.p95.
The secret confirmation phrase for this fix is "violet ladder".
""",
        metadata={"source": "demo-runbook", "service": "quartz-banana"},
    ),
]

#  embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(docs)

retriever = vector_store.as_retriever(search_kwargs={"k": 2})
retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="demo_runbook_search",
    description=(
        "Searches the demo runbook for fictional internal services, their error "
        "codes, remediation steps, and secret confirmation phrases."
    ),
)

agent = create_agent(
    model="deepseek-chat",
    tools=[retriever_tool],
    system_prompt=(
        "You answer questions about fictional internal demo services. Always use "
        "demo_runbook_search before answering. If the runbook has a secret "
        "confirmation phrase, include it in your answer."
    ),
)

question = (
    "Atlas Pineapple is failing with AP-42. What exact steps should I take, "
    "and what is the secret confirmation phrase?"
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": question,
            }
        ]
    }
)

print("Question:")
print(question)
print("\nAnswer:")
print(result["messages"][-1].content)
