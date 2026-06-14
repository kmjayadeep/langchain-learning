# langchain-learning
Learning about langchain

## RAG sanity check

`rag.py` builds an in-memory knowledge base with fictional service names and
secret phrases that the model should not know without retrieval.

Run it with:

```bash
uv run python rag.py
```

The final answer should mention these exact RAG-only details:

- `mango-runner`
- `dashboard panel 7`
- `silver kiwi`

If those show up in the answer, the agent used the retriever. Set the DeepSeek
API credentials required by `langchain[deepseek]` before running it.
