from langchain_google_genai import GoogleGenerativeAIEmbeddings
import numpy as np

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview", output_dimensionality=768)

def basic_embeddings():
    text = "LangChain is a framework for developing applications powered by language models."
    embedding = embeddings.embed_query(text)
    
    print(f"Embedding dimensions: {len(embedding)}")
    print(f"First 5 values of the embedding: {embedding[:5]}")
    print(f"Vector norm: {sum(x**2 for x in embedding) ** 0.5}")


def batch_embeddings():
    texts = [
        "LangChain is a framework for developing applications powered by language models.",
        "It was created by Jd in October 2030.",
        "LangGraph is a library for building stateful, multi-actor applications."
    ]
    embeddings_batch = embeddings.embed_documents(texts)
    
    for i, emb in enumerate(embeddings_batch):
        print(f"Text {i+1}: {texts[i]}")
        print(f"Embedding dimensions: {len(emb)}")
        print(f"First 5 values of the embedding: {emb[:5]}")
        print(f"Vector norm: {sum(x**2 for x in emb) ** 0.5}")

def similarity_search():
    docs = [
        "Python is a high-level programming language.",
        "Java is a widely-used programming language.",
        "LangChain is a framework for developing applications powered by language models.",
        "cat is an animal.",
    ]

    query = "what programming languages exist?"

    doc_embeddings = embeddings.embed_documents(docs)
    query_embedding = embeddings.embed_query(query)

    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


    similarities = [cosine_similarity(query_embedding, doc_emb) for doc_emb in doc_embeddings]
    ranked_docs = sorted(zip(docs, similarities), key=lambda x: x[1], reverse=True)

    print(f"Query: {query}")
    print("Ranked documents based on similarity:")
    for doc, score in ranked_docs:
        print(f"Document: {doc}, Similarity Score: {score:.4f}")


#  basic_embeddings()
#  batch_embeddings()

similarity_search()
