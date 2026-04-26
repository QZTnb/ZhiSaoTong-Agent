from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uuid
from agent.langgraph_agent import MainAgent

app = FastAPI(
    title="智扫通智能机器人客服API",
    description="基于LangGraph+MCP的智能客服系统，支持扫地机器人售前/售后/运维全场景",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化Agent
agent = MainAgent()

# 定义请求模型
class ChatRequest(BaseModel):
    query: str
    session_id: str = None

# 定义响应模型
class ChatResponse(BaseModel):
    response: str
    session_id: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """智能客服对话接口"""
    try:
        # 如果没有提供session_id，生成一个新的
        session_id = request.session_id or str(uuid.uuid4())
        
        # 执行查询
        print(f"Executing query: {request.query}")
        response = agent.execute(request.query, session_id)
        print(f"Response: {response}")
        
        return ChatResponse(response=response, session_id=session_id)
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"内部错误: {str(e)}")

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """智能客服流式对话接口"""
    try:
        # 如果没有提供session_id，生成一个新的
        session_id = request.session_id or str(uuid.uuid4())
        
        # 流式执行查询
        def generate():
            for chunk in agent.execute_stream(request.query, session_id):
                yield f"data: {chunk}\n\n"
        
        return StreamingResponse(generate(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"内部错误: {str(e)}")

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    # 使用硬编码的端口号8014
    uvicorn.run(app, host="0.0.0.0", port=8014)
