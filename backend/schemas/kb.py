from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    session_id: str = Field(default="default", description="会话ID，区分不同对话")


class UploadResponse(BaseModel):
    filename: str
    chunks_count: int
