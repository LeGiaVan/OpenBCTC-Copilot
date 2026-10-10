import json
import asyncio
from fastapi import APIRouter, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from src.agents.copilot.graph import run_query

router = APIRouter(tags=["Chat"])

class ChatRequest(BaseModel):
    query: str
    company: str = "VNM"
    year: int = 2025
    thread_id: str = "web-session-1"

@router.post("/chat")
async def chat_endpoint(request: Request, body: ChatRequest):
    graph = request.app.state.graph
    
    # Chạy đồng bộ run_query trong executor để không block event loop
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, 
        lambda: run_query(
            graph=graph,
            query=body.query,
            company=body.company,
            year=body.year,
            thread_id=body.thread_id
        )
    )
    
    # SSE Generator để giả lập streaming (cho cảm giác gõ từng chữ trên UI)
    async def sse_generator():
        import re
        tokens = re.findall(r'\S+|\s+', result["final_answer"])
        for token in tokens:
            yield {
                "event": "token",
                "data": json.dumps({"token": token}, ensure_ascii=False)
            }
            await asyncio.sleep(0.005)  # Giả lập streaming delay mượt mà
        
        # Gửi metadata citations & status ở sự kiện cuối cùng
        meta = {
            "intent": result["intent"].value if result["intent"] else None,
            "fact_check": result["fact_check_status"].value if result["fact_check_status"] else None,
            "citations": result["citations"]
        }
        yield {"event": "metadata", "data": json.dumps(meta, ensure_ascii=False)}
        yield {"event": "done", "data": "[DONE]"}
        
    return EventSourceResponse(sse_generator())
