import os
import fitz  # pymupdf
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config.ai_conf import (
    EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_DIR,
)

# 全局向量库实例
vectorstore: Chroma = None

embeddings = OpenAIEmbeddings(
    api_key=EMBEDDING_API_KEY,
    base_url=EMBEDDING_BASE_URL,
    model=EMBEDDING_MODEL,
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", "。", ".", " "],
)


def _extract_text_from_pdf(file_path: str) -> str:
    """从 PDF 中提取所有文字，逐页拼接。扫描件 PDF 可能返回空字符串。"""
    doc = fitz.open(file_path)
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    return "\n".join(text_parts)


def ingest_file(file_path: str) -> tuple[int, list[str]]:
    """加载 txt/pdf 文件 → 切分 → 向量化 → 存入 Chroma，返回 (chunk数, 文档ID列表)"""
    global vectorstore

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        text = _extract_text_from_pdf(file_path)
        if not text.strip():
            raise ValueError("PDF 中未提取到文字，可能是扫描件（图片PDF）")
        docs = [Document(page_content=text, metadata={"source": file_path})]
    else:
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()

    chunks = splitter.split_documents(docs)

    if vectorstore is None:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DIR,
        )
    else:
        vectorstore.add_documents(chunks)

    doc_ids = [chunk.metadata.get("source", "") for chunk in chunks]
    return len(chunks), doc_ids


def get_retriever():
    """获取检索器，如果还没初始化就尝试从磁盘加载"""
    global vectorstore
    if vectorstore is None:
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        )
    return vectorstore.as_retriever(search_kwargs={"k": 3})


def delete_document(file_path: str):
    """从向量库中删除指定文档的所有切片"""
    global vectorstore
    if vectorstore is None:
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        )
    vectorstore.delete(where={"source": file_path})
