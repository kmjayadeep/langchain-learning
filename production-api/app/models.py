from pydantic import BaseModel, Field
from datetime import datetime


class ChatRequest(BaseModel):
    message: str = Field(
        min_length = 1,
        max_length=10000,
        description="The user's message to the agent"
    )
    thread_id: str = Field(
        default="default",
        description="Conversation thread id"
    )

class ChatResponse(BaseModel):
    response: str
    thread_id: str
    model_used: str
    cached: bool = False
    processing_time_ms: float
    timestamp: str = Field(default_factory=lambda: datetime.now())

    class HealthResponse(BaseModel):
        status: str = "healthy"
        environment: str
        version: str = "1.0.0"
        checks: dict = {}
