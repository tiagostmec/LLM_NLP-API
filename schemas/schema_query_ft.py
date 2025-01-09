from pydantic import BaseModel, field_validator
from typing import Literal

ROUTE = "query_from_fine_tuning"

class QueryFtRequest(BaseModel):
    user_message: str  #  user text

class QueryFtResponse(BaseModel):
    message: str


