import os
from langchain_core.tools import tool
from services.doc_service import get_retriever


@tool
def search_knowledge_base(query: str) -> str:
    """搜索知识库获取相关文档内容。当用户询问需要查文档的问题时调用此工具。"""
    retriever = get_retriever()
    docs = retriever.invoke(query)
    if not docs:
        return "知识库中未找到相关信息。"
    return "\n\n".join(
        f"[来源{i + 1}]\n{d.page_content}" for i, d in enumerate(docs)
    )


@tool
def list_knowledge_base_documents() -> str:
    """列出知识库中已上传的所有文档（文件名和数量）。当用户询问上传了哪些文档、有多少文档、文档列表时调用此工具。"""
    uploads_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "uploads",
    )
    if not os.path.exists(uploads_dir):
        return "知识库中暂无文档（uploads 目录不存在）。"

    files = sorted([
        f for f in os.listdir(uploads_dir)
        if f.endswith((".txt", ".pdf"))
    ])

    if not files:
        return "知识库中暂无文档。"

    file_list = "\n".join(f"  - {f}" for f in files)
    return f"知识库中共有 {len(files)} 个文档：\n{file_list}"


