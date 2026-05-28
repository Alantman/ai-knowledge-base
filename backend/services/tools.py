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
