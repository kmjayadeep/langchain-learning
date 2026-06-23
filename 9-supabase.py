import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document

embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

DB_URL = os.getenv("DB_URL")


docs = [
    Document(
        page_content="there are cats in the pond",
        metadata={"id": 1, "location": "pond", "topic": "animals"},
    ),
    Document(
        page_content="ducks are also found in the pond",
        metadata={"id": 2, "location": "pond", "topic": "animals"},
    ),
    Document(
        page_content="fresh apples are available at the market",
        metadata={"id": 3, "location": "market", "topic": "food"},
    ),
    Document(
        page_content="the market also sells fresh oranges",
        metadata={"id": 4, "location": "market", "topic": "food"},
    ),
    Document(
        page_content="the new art exhibit is fascinating",
        metadata={"id": 5, "location": "museum", "topic": "art"},
    ),
    Document(
        page_content="a sculpture exhibit is also at the museum",
        metadata={"id": 6, "location": "museum", "topic": "art"},
    ),
    Document(
        page_content="a new coffee shop opened on Main Street",
        metadata={"id": 7, "location": "Main Street", "topic": "food"},
    ),
    Document(
        page_content="the book club meets at the library",
        metadata={"id": 8, "location": "library", "topic": "reading"},
    ),
    Document(
        page_content="the library hosts a weekly story time for kids",
        metadata={"id": 9, "location": "library", "topic": "reading"},
    ),
    Document(
        page_content="a cooking class for beginners is offered at the community center",
        metadata={"id": 10, "location": "community center", "topic": "classes"},
    ),
]



def connect_db():

    vectorstore = PGVector(
        embeddings = embeddings_model,
        collection_name = "production_docs",
        connection=DB_URL,
        use_jsonb=True,
    )

    return vectorstore


def test_db():
    store = connect_db()
    store.add_documents(docs, ids=[doc.metadata["id"] for doc in docs])

    results = store.similarity_search(
        "kitty", k=10, filter={"id": {"$in": [1, 5, 2, 9]}}
    )
    for doc in results:
        print(f"* {doc.page_content} [{doc.metadata}]")


test_db()
