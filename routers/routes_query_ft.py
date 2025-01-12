from fastapi_utils.cbv import cbv
from fastapi_utils.inferring_router import InferringRouter
from schemas.schema_query_ft import QueryFtRequest, QueryFtResponse
from fastapi.responses import JSONResponse
from services.query_ft import query_ft


router = InferringRouter()

@cbv(router)
class FineTuning: 
    @router.post("/query_from_fine_tuning",
        description="Performs the use of trained finetuned data.",
        response_description="Returns inference message",
        response_model=QueryFtResponse,
    )
    async def ask_question(self, data: QueryFtRequest):
        response_data = query_ft(
            msg = data.user_message
        )   
        return JSONResponse(status_code=200, content=response_data)