from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    output_dimensionality=768,
)

document = """
# Platform Support Notes

Developers report many different problems to the platform team. Some are runtime failures, some are access requests, and some are release blockers.

The checkout service failed after deployment. The pod started successfully but exited a few seconds later. The previous container logs mention a missing DATABASE_URL value. This points to a Kubernetes application configuration problem, not a GitHub Actions runner problem.

The payment workflow did not start. The GitHub Actions job stayed queued for several minutes. The job required the linux-large runner label, but no online self-hosted runner matched that label. This points to a CI runner availability or label configuration issue.

A developer asked for Artifactory access because their build was blocked. This should not be treated as a CI failure. The platform team should check the requester, target repository, permission target, owner approval, and whether temporary access is enough.

The release was blocked during publishing. The releasability report passed, but artifact signing failed before the package was uploaded. This points to the signing service, certificate validity, or publishing credentials.
"""


def recursive_store():
    # Intentionally small so recursive splitting cuts concepts apart.
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=180,
        chunk_overlap=0,
        separators=["\n\n", ". ", "\n", " "],
    )

    recursive_chunks = recursive_splitter.split_text(document)

    print("\n\n===== RECURSIVE CHUNKS =====")
    for i, chunk in enumerate(recursive_chunks):
        print(f"\n-- Chunk {i + 1} ({len(chunk)} chars) ---")
        print(chunk)

    recursive_vectorstore = Chroma.from_texts(
        recursive_chunks,
        embeddings,
        collection_name="recursive_chunks",
    )

    return recursive_vectorstore


def semantic_store():
    semantic_chunker = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=75,
    )

    semantic_chunks = semantic_chunker.split_text(document)
    semantic_chunks = [chunk for chunk in semantic_chunks if chunk.strip()]

    print("\n\n===== SEMANTIC CHUNKS =====")
    for i, chunk in enumerate(semantic_chunks):
        print(f"\n-- Chunk {i + 1} ({len(chunk)} chars) ---")
        print(chunk)

    semantic_vectorstore = Chroma.from_texts(
        semantic_chunks,
        embeddings,
        collection_name="semantic_chunks",
    )

    return semantic_vectorstore


def test_retrieval(query, vectorstore, name):
    results = vectorstore.similarity_search(query, k=1)

    print(f"\n===== {name} =====")
    print(f"Query: {query}")
    print("Top result:")
    print(results[0].page_content)
    print("=" * 40)

    return results[0].page_content


rs = recursive_store()
ss = semantic_store()

queries = [
    "missing database url",
    "runner label mismatch",
    "artifactory approval",
    "artifact signing failed",
]

for query in queries:
    print("\n" + "==" * 60)
    test_retrieval(query, rs, "RECURSIVE")
    test_retrieval(query, ss, "SEMANTIC")
