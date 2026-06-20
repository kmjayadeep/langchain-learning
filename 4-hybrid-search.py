from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_chroma import Chroma

embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")


documents = [
        Document(
            page_content="SKU12345 is a red t-shirt made of 100% cotton. It is available in sizes S, M, L, and XL.",
            metadata={"source": "product_catalog"}
            ),
        Document(
            page_content="SKU12345 is out of stock now",
            metadata={"source": "product_catalog"}
            ),
        Document(
            page_content="The red t-shirt (SKU12345) is currently out of stock. Expected restock date is next month.",
            metadata={"source": "inventory_system"}
            ),
        Document(
            page_content="For network connectivity issues, try restarting your router and checking your cables. If the problem persists, contact your ISP.",
            metadata={"source": "troubleshooting_guide"}
            ),
        Document(
            page_content="The authentication service is currently experiencing high latency. Our team is investigating the issue and working on a fix.",
            metadata={"source": "status_updates"}
            ),
]  


vectorstore = Chroma.from_documents(
        documents,
        embeddings_model,
        collection_name="hybrid_test",
)

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

bm25_retriever = BM25Retriever.from_documents(documents, k=3)

ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5]
)

def test_query(query, name, retriever):
    print(f"\n{name} results for query: '{query}'")
    results = retriever.invoke(query)
    for i, doc in enumerate(results):
        print(f"Result {i+1}: {doc.page_content} (Source: {doc.metadata['source']})")

    return results

test_queries = [
    "What is the status of SKU12345?",
    "How can I troubleshoot network issues?",
    "What is the current status of the authentication service?"
]

for query in test_queries:
    print('=' * 50)
    vector_results = test_query(query, "Vector", vector_retriever)
    bm25_results = test_query(query, "BM25", bm25_retriever)
    ensemble_results = test_query(query, "Ensemble", ensemble_retriever)

