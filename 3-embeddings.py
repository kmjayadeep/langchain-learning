from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview", output_dimensionality=768)

def basic_embeddings():
    text = "LangChain is a framework for developing applications powered by language models."
    embedding = embeddings.embed_query(text)
    
    print(f"Embedding dimensions: {len(embedding)}")
    print(f"First 5 values of the embedding: {embedding[:5]}")
    print(f"Vector norm: {sum(x**2 for x in embedding) ** 0.5}")

basic_embeddings()
