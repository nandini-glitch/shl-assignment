"""Request/response models. The shape here is non-negotiable per the spec:
POST /chat takes a stateless message list and returns {reply, recommendations,
end_of_conversation}. recommendations is [] when the agent isn't presenting
a shortlist this turn (clarifying, comparing, refusing) and 1-10 items when
it is.
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)

    @field_validator("messages")
    @classmethod
    def must_end_with_user(cls, v: list[ChatMessage]) -> list[ChatMessage]:
        if v[-1].role != "user":
            raise ValueError("the last message in the conversation must be from the user")
        return v


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False


class HealthResponse(BaseModel):
    status: str = "ok"
