from fastapi import FastAPI, HTTPException
from langchain.messages import HumanMessage
from pydantic import BaseModel
import uvicorn
# Import your existing LangGraph logic here (agent_app, vector_store)
from app import agent_app, vector_store 

app = FastAPI(title="Sarthak's Portfolio API")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # This allows your Streamlit UI to talk to FastAPI
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def health_check():
    """This endpoint tells the frontend that the backend is alive."""
    return {"status": "online"}

class ChatRequest(BaseModel):
    question: str
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    config = {"configurable": {"thread_id": "session_1"}}
    
    # Notice: NO "question" key here. We use "messages".
    initial_input = {
        "messages": [HumanMessage(content=request.question)], # 'request.question' comes from your Pydantic model
        "context": "",
        "count": 0,
        "steps": []
    }
    
    try:
        # Use ainvoke for FastAPI
        result = await agent_app.ainvoke(initial_input, config=config)
        return {
            "answer": result["answer"],
            "count": result.get("count", 0)
        }
    except Exception as e:
        print(f"ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import os
    # Render provides a $PORT environment variable. If it's missing, use 10000.
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
