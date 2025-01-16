from pydantic import BaseModel, field_validator

ROUTE = "rag_openai"

class RagOpenAIRequest(BaseModel):
    anime_name: str

class RagOpenAIResponse(BaseModel):
    message: str


#Add here field validators in the future