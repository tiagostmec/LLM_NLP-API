from pydantic import BaseModel, field_validator

ROUTE = "rag_hf"

class RagHfRequest(BaseModel):
    anime_name: str

class RagHfResponse(BaseModel):
    message: str



#Add here field validators in the future