import os
import httpx
import fitz  # pymupdf
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config.ai_conf import (
    EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_DIR,
)


class SiliconFlowEmbeddings(Embeddings):
    """
    自定义硅基流动 Embedding 封装类。
    不使用 OpenAI SDK，直接用 httpx 调硅基流动 API，
    避免 OpenAI SDK 把中文 tokenize 成 token IDs 导致硅基流动 500 错误。
    """
    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.Client(timeout=60)

    def embed_query(self, text: str) -> List[float]:
        """向量化单条查询文本"""
        return self._embed([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """向量化多条文档文本"""
        return self._embed(texts)

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """调用硅基流动 Embedding API，直接发送原始文本"""
        resp = self._client.post(
            f"{self.base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.model, "input": texts},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Embedding API error {resp.status_code}: {resp.text}")
        data = resp.json()
        # 按索引排序后返回 embedding 向量列表
        sorted_items = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_items]


# 全局向量库实例
vectorstore: Chroma = None

embeddings = SiliconFlowEmbeddings(
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
