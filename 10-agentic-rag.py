import os
import re
from typing import Literal, TypedDict

# LangSmith tracing: set LANGSMITH_API_KEY in your shell for traces to upload.
os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")  # older LangChain env var
os.environ.setdefault("LANGSMITH_PROJECT", "langchain-learning-agentic-rag")

from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langgraph.graph import END, StateGraph
from langsmith import traceable

# Pre-chunked sample knowledge base.
# The chunks intentionally mix related and unrelated topics so the grader has work to do.
KB_DOCS = [
    Document(
        page_content=(
            "Product glossary: Mercury is Acme's invoice and payments portal. "
            "Nimbus is the analytics warehouse; some teams call it the sky warehouse. "
            "Helios is the customer support escalation assistant."
        ),
        metadata={"source": "product_glossary.md", "topic": "glossary"},
    ),
    Document(
        page_content=(
            "Mercury invoice runbook: If an invoice PDF is blank, first click Regenerate PDF. "
            "If it is still blank after 10 minutes, clear the browser cache and retry. "
            "If the invoice remains blank, open a P2 ticket with the billing-ops team."
        ),
        metadata={"source": "mercury_runbook.md", "topic": "billing"},
    ),
    Document(
        page_content=(
            "Mercury approvals: invoices over $5,000 require manager approval. "
            "Approvals must be completed by the 25th of each month to avoid payment delay. "
            "Finance owns approval policy questions."
        ),
        metadata={"source": "mercury_approvals.md", "topic": "billing"},
    ),
    Document(
        page_content=(
            "Nimbus cost controls: To reduce analytics warehouse spend, archive tables unused "
            "for 90 days, downsample event logs older than 30 days, and set dashboard refreshes "
            "to hourly instead of every 5 minutes."
        ),
        metadata={"source": "nimbus_costs.md", "topic": "analytics"},
    ),
    Document(
        page_content=(
            "Nimbus freshness SLA: standard dashboards refresh every hour. Executive dashboards "
            "refresh every 15 minutes. Backfills should be scheduled after 8 PM UTC."
        ),
        metadata={"source": "nimbus_sla.md", "topic": "analytics"},
    ),
    Document(
        page_content=(
            "Helios escalation policy: Escalate to Sev-2 when a paying customer cannot complete "
            "checkout, when more than 20 users are blocked, or when an issue has no workaround "
            "for over 30 minutes."
        ),
        metadata={"source": "helios_escalation.md", "topic": "support"},
    ),
    Document(
        page_content=(
            "Office Wi-Fi help: The guest network is Acme-Guest. The password changes every Monday. "
            "Employees should use Acme-Secure with single sign-on."
        ),
        metadata={"source": "office_wifi.md", "topic": "office"},
    ),
    Document(
        page_content=(
            "Travel expense policy: meal receipts are required for expenses over $25. "
            "Flights should be booked at least 14 days in advance when possible."
        ),
        metadata={"source": "travel_policy.md", "topic": "hr"},
    ),
]


class RagState(TypedDict):
    query: str
    rewritten_query: str
    documents: list[Document]
    generation: str
    relevance_score: float
    retry_count: int
    max_retries: int


embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
llm = init_chat_model(
    model="deepseek-v4-flash",
    temperature=0.2,
)

_vectorstore = None


@traceable(name="create_knowledge_base")
def create_kb():
    return Chroma.from_documents(
        documents=KB_DOCS,
        embedding=embeddings_model,
        collection_name="agentic-rag",
    )


@traceable(name="retrieve_documents")
def retrieve_documents(state: RagState) -> dict:
    """Retrieve documents using the rewritten query if we have one."""
    global _vectorstore

    query = state.get("rewritten_query") or state["query"]
    print("Searching for:", query)

    if _vectorstore is None:
        _vectorstore = create_kb()

    retriever = _vectorstore.as_retriever(search_kwargs={"k": 3})
    documents = retriever.invoke(query)

    print(f"Retrieved {len(documents)} document(s)")
    for i, doc in enumerate(documents, 1):
        print(f"Doc {i} ({doc.metadata['source']}): {doc.page_content[:120]}...\n")

    return {"documents": documents}


def _parse_score(text: str) -> float:
    """Extract a 0-1 score from model output."""
    matches = re.findall(r"\d+(?:\.\d+)?", text)
    if not matches:
        return 0.0
    return max(0.0, min(1.0, float(matches[-1])))


@traceable(name="grade_documents")
def grade_documents(state: RagState) -> dict:
    """Ask the LLM how relevant each document is."""
    query = state["query"]
    documents = state["documents"]

    grading_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You grade if a document is useful for answering a question. "
                "Return only one number from 0 to 1.",
            ),
            ("human", "Question: {query}\n\nDocument:\n{document}\n\nScore:"),
        ]
    )

    chain = grading_prompt | llm
    relevant_docs = []
    relevant_scores = []

    for doc in documents:
        result = chain.invoke({"query": query, "document": doc.page_content})
        score = _parse_score(result.content)

        source = doc.metadata.get("source", "unknown")
        if score >= 0.5:
            print(f"Keeping {source} with score {score:.2f}")
            relevant_docs.append(doc)
            relevant_scores.append(score)
        else:
            print(f"Dropping {source} with score {score:.2f}")

    avg_score = sum(relevant_scores) / len(relevant_scores) if relevant_scores else 0.0
    print(f"Average relevance score: {avg_score:.2f}")

    return {"documents": relevant_docs, "relevance_score": avg_score}


@traceable(name="rewrite_query")
def rewrite_query(state: RagState) -> dict:
    """Rewrite the question to make retrieval easier."""
    retry_count = state.get("retry_count", 0)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Rewrite the user question as a short search query. "
                "Keep the meaning the same.",
            ),
            ("human", "Original question: {query}\nSearch query:"),
        ]
    )

    chain = prompt | llm
    result = chain.invoke({"query": state["query"]})
    rewritten = result.content.strip()

    print("Rewritten query:", rewritten)
    return {"rewritten_query": rewritten, "retry_count": retry_count + 1}


@traceable(name="generate_answer")
def generate_answer(state: RagState) -> dict:
    query = state["query"]
    documents = state["documents"]

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
        for doc in documents
    )

    generate_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer using only the context. If the context does not contain "
                "the answer, say you do not know. Keep it concise.",
            ),
            ("human", "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"),
        ]
    )

    chain = generate_prompt | llm
    result = chain.invoke({"context": context, "query": query})

    return {"generation": result.content}


@traceable(name="generate_fallback")
def generate_fallback(state: RagState) -> dict:
    """Fallback response when retrieval fails after all retries."""
    fallback_message = (
        "I could not find enough relevant information in the knowledge base "
        "to answer that question."
    )
    return {"generation": fallback_message}


## Routing


def should_retry_or_generate(state: RagState) -> Literal["rewrite", "generate", "fallback"]:
    """Decide what the graph should do next."""
    has_docs = len(state.get("documents", [])) > 0
    good_score = state.get("relevance_score", 0.0) >= 0.5
    can_retry = state.get("retry_count", 0) < state.get("max_retries", 1)

    if has_docs and good_score:
        return "generate"
    if can_retry:
        return "rewrite"
    return "fallback"


def build_graph():
    workflow = StateGraph(RagState)

    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("grade", grade_documents)
    workflow.add_node("rewrite", rewrite_query)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("fallback", generate_fallback)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges(
        "grade",
        should_retry_or_generate,
        {
            "rewrite": "rewrite",
            "generate": "generate",
            "fallback": "fallback",
        },
    )
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", END)
    workflow.add_edge("fallback", END)

    return workflow.compile()


def print_graph():
    """Print a simple view of the graph edges."""
    print("""
Graph:

START
  ↓
retrieve_documents
  ↓
grade_documents
  ├─ relevant docs found       → generate_answer → END
  ├─ not relevant, can retry   → rewrite_query → retrieve_documents
  └─ not relevant, no retries  → generate_fallback → END
""")


def run_demo():
    graph = build_graph()
    print_graph()

    questions = [
        # Direct hit: retrieval should find the Mercury runbook.
        "My Mercury invoice PDF is blank. What should I do?",
        # Mixed hit: retrieval may bring glossary + Nimbus docs; grading should keep useful docs.
        "For the sky warehouse, how do we lower storage spend?",
        # Direct hit: support policy question.
        "When should Helios escalate an issue to Sev-2?",
        # No good answer: retrieved docs should be graded low, retried once, then fallback.
        "What is the cafeteria lunch menu today?",
    ]

    for question in questions:
        print("\n" + "=" * 80)
        print("Question:", question)

        result = graph.invoke(
            {
                "query": question,
                "rewritten_query": "",
                "documents": [],
                "generation": "",
                "relevance_score": 0.0,
                "retry_count": 0,
                "max_retries": 1,
            },
            config={
                "run_name": "agentic_rag_demo",
                "tags": ["agentic-rag", "learning"],
                "metadata": {"question": question},
            },
        )

        print("Answer:", result["generation"])


if __name__ == "__main__":
    run_demo()
