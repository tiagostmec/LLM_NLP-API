from fastapi_utils.cbv import cbv
from fastapi_utils.inferring_router import InferringRouter
from schemas.schema_rag_hf import RagHfRequest, RagHfResponse
from fastapi.responses import JSONResponse
from services.rag_try import answer_anime_question_rag


router = InferringRouter()

@cbv(router)
class RAGOpenAI: 
    @router.post("/rag_hf",
        description="Performs the RAG with BART for a llm model with api jikan focus.",
        response_description="Returns a message from sucess",
        response_model=RagHfResponse,
    )
    async def answer_rag_hf(self, data: RagHfRequest):
        response_data = answer_anime_question_rag(
            anime_name = data.anime_name,
        )   
        return JSONResponse(status_code=200, content=response_data)
  
