from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import kb

app = FastAPI(title="AI 知识库问答系统")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(kb.router)


@app.get("/")
async def root():
    return {"message": "AI 知识库问答系统"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8010)
