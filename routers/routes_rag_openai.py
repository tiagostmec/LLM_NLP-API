from fastapi_utils.cbv import cbv
from fastapi_utils.inferring_router import InferringRouter
from schemas.schema_rag_openai import RagOpenAIRequest, RagOpenAIResponse
from fastapi.responses import JSONResponse
from services.rag_try_openai_api import answer_anime_question_gpt3


router = InferringRouter()

@cbv(router)
class RAGOpenAI: 
    @router.post("/rag_openai",
        description="Performs the RAG for a llm model with api OPENAI and jikan focus.",
        response_description="Returns a message from sucess",
        response_model=RagOpenAIResponse,
    )
    async def answer_rag_openai(self, data: RagOpenAIRequest):
        response_data = answer_anime_question_gpt3(
            anime_name=data.anime_name
        )   
        return JSONResponse(status_code=200, content=response_data)
  
