from pydantic import BaseModel, field_validator
from typing import Literal

ROUTE = "fine_tuning_gpt"

class FineTuneRequest(BaseModel):
    csv_file: str  # path to csv data
    model_type: Literal['bert', 'gpt2']

class FineTuneResponse(BaseModel):
    message: str


