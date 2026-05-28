import os
import asyncio
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from schemas.kb import ChatRequest
from services.doc_service import ingest_file, _extract_text_from_pdf, delete_document
from services.rag_service import ask_stream, ask_agent_stream, ask_agent_reasoning_stream, ask_with_file_stream
from utils.response import success_response, error_response

router = APIRouter(prefix="/api/kb", tags=["知识库问答"])

UPLOADS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "uploads")


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    """上传 .txt 文件，自动切分并存入向量库"""
    if not file.filename.endswith((".txt", ".pdf")):
        return error_response("仅支持 .txt 和 .pdf 文件")

    # 保存到 uploads/
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    file_path = os.path.join(UPLOADS_DIR, file.filename)
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # 切分 + 向量化 + 存库
    try:
        chunks_count, _ = ingest_file(file_path)
    except Exception as e:
        return error_response(f"文档处理失败：{str(e)}")

    return success_response(
        message=f"上传成功，切分为 {chunks_count} 个片段",
        data={"filename": file.filename, "chunks_count": chunks_count},
    )


@router.get("/documents")
async def list_documents():
    """列出已上传的文档"""
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    files = [f for f in os.listdir(UPLOADS_DIR) if f.endswith((".txt", ".pdf"))]
    return success_response(data={"documents": files})


@router.delete("/documents/{filename}")
async def remove_document(filename: str):
    """删除已上传的文档及对应的向量数据"""
    file_path = os.path.join(UPLOADS_DIR, filename)
    if not os.path.exists(file_path):
        return error_response("文件不存在")

    try:
        delete_document(file_path)
    except Exception as e:
        return error_response(f"删除向量数据失败：{str(e)}")

    os.remove(file_path)
    return success_response(message=f"已删除 {filename}")


@router.post("/chat")
async def chat(req: ChatRequest):
    """流式问答，固定先检索再回答"""
    async def generate():
        try:
            for chunk in ask_stream(req.question, req.session_id):
                yield chunk
                await asyncio.sleep(0.08)  # 模拟网络延迟让流式可见
        except Exception as e:
            yield f"\n[错误：{str(e)}]"

    return StreamingResponse(
        generate(),
        media_type="text/plain; charset=utf-8",
    )


@router.post("/chat/agent")
async def chat_agent(req: ChatRequest):
    """流式问答，模型自主决定是否搜索知识库（单轮 Tool Calling）"""
    async def generate():
        try:
            for chunk in ask_agent_stream(req.question, req.session_id):
                yield chunk
                await asyncio.sleep(0.05)
        except Exception as e:
            yield f"\n[错误：{str(e)}]"

    return StreamingResponse(
        generate(),
        media_type="text/plain; charset=utf-8",
    )


@router.post("/chat/agent/reasoning")
async def chat_agent_reasoning(req: ChatRequest):
    """流式问答，Agent 多轮推理：模型可以连续多次调用工具"""
    async def generate():
        try:
            for chunk in ask_agent_reasoning_stream(req.question, req.session_id):
                yield chunk
                await asyncio.sleep(0.08)
        except Exception as e:
            yield f"\n[错误：{str(e)}]"

    return StreamingResponse(
        generate(),
        media_type="text/plain; charset=utf-8",
    )


@router.post("/chat/with-file")
async def chat_with_file(
    file: UploadFile = File(...),
    question: str = Form(...),
    session_id: str = Form(default="default"),
):
    """临时文件分析：上传文件 → 提取文字 → 拼入 prompt → 流式回复（不入库）"""
    if not file.filename.endswith((".txt", ".pdf")):
        return error_response("仅支持 .txt 和 .pdf 文件")

    os.makedirs(UPLOADS_DIR, exist_ok=True)
    tmp_path = os.path.join(UPLOADS_DIR, f"_tmp_{file.filename}")

    content = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(content)

    ext = os.path.splitext(file.filename)[1].lower()
    if ext == ".pdf":
        text = _extract_text_from_pdf(tmp_path)
    else:
        with open(tmp_path, "r", encoding="utf-8") as f:
            text = f.read()

    os.remove(tmp_path)

    if not text.strip():
        return error_response("文件中未提取到文字，可能是扫描件 PDF")

    async def generate():
        try:
            for chunk in ask_with_file_stream(text, question, session_id):
                yield chunk
                await asyncio.sleep(0.08)
        except Exception as e:
            yield f"\n[错误：{str(e)}]"

    return StreamingResponse(
        generate(),
        media_type="text/plain; charset=utf-8",
    )
