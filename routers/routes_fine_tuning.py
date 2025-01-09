from fastapi_utils.cbv import cbv
from fastapi_utils.inferring_router import InferringRouter
from schemas.schema_fine_tuning import FineTuneRequest, FineTuneResponse
from fastapi.responses import JSONResponse
from services.fine_tuning import fine_tuning


router = InferringRouter()

@cbv(router)
class FineTuning: 
    @router.post("/fine_tuning",
        description="Performs the fine tuning for a gpt or bert with stritc sent data.",
        response_description="Returns a message from sucess",
        response_model=FineTuneResponse,
    )
    async def fine_tune(self, data: FineTuneRequest):
        response_data = await fine_tuning(
            csv_file = data.csv_file,
            model_type =  data.model_type,
            output_dir= 'C:\\Users\\th_sm\\Desktop\\LLM_NLP API\\data'
        )   
        return JSONResponse(status_code=200, content=response_data)
  
