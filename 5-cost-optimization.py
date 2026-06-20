from langchain.chat_models import init_chat_model
from langsmith import traceable


class TokenBudget:

    def __init__(self, max_tokens_per_request: int = 1000):
        self.max_per_request = max_tokens_per_request
        self.usage = {
            "total_input": 0,
            "total_output": 0,
            "requests": 0
        }

    def estimate_tokens(self, text: str) -> int:
        return int(len(text.split()) * 1.3)

    def check_budget(self, text: str) -> tuple[bool, int]:
        tokens = self.estimate_tokens(text)
        return tokens <= self.max_per_request, tokens

    def record_usage(self, input_tokens: int, output_tokens:int):
        self.usage["total_input"]+= input_tokens
        self.usage["total_output"]+= output_tokens
        self.usage["requests"] +=1

    def get_stats(self) -> dict:
        return {
            **self.usage,
            "total_tokens": self.usage["total_input"] + self.usage["total_output"],
            "avg_per_request": (
                (self.usage["total_input"] + self.usage["total_output"]) 
                / max(self.usage["requests"], 1)
            ),
        }

class BudgetedLLM:

    def __init__(self, max_tokens: int = 4000):
        self.budget = TokenBudget(max_tokens_per_request=max_tokens)
        self.llm = init_chat_model(
            model='deepseek-v4-flash',
            temperature=0.7,
        )

    @traceable(name="budgeted_invoke")
    def invoke(self, query: str) -> str:

        within_budget, tokens = self.budget.check_budget(query)

        if not within_budget:
            raise ValueError(
                f"Query exceeds token budget: {tokens} > {self.budget.max_per_request}"
            )
    
        res = self.llm.invoke(query)
        result = res.content

        output_tokens = self.budget.estimate_tokens(result)
        self.budget.record_usage(tokens, output_tokens)

        return result

    def get_stats(self) -> dict:
        return self.budget.get_stats()


def demo_token_budgetting():

    llm = BudgetedLLM(max_tokens=100)

    queries = [
        "What is AI?",
        "Explain" + "very " *100 + "complex topic"
    ]

    print("token budgetting demo")

    for query in queries:
        try:
            result = llm.invoke(f"Answer briefly, in under 50 words:\n\n{query}")
            print(f"✅ {query[:40]}... -> {result[:30]}...")
        except ValueError as e:
            print(f"❌ {query[:40]}... -> {e}")

    print(f"\nUsage: {llm.get_stats()}")


demo_token_budgetting()
