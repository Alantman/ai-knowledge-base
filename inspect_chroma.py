"""检查 Chroma 向量库内容"""
import sys
sys.path.insert(0, 'backend')

from config.ai_conf import CHROMA_DIR, EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_MODEL
from services.doc_service import SiliconFlowEmbeddings
from langchain_chroma import Chroma

embeddings = SiliconFlowEmbeddings(
    api_key=EMBEDDING_API_KEY,
    base_url=EMBEDDING_BASE_URL,
    model=EMBEDDING_MODEL,
)

vs = Chroma(
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)

collection = vs._collection
count = collection.count()

print(f"Total chunks: {count}")
print()

if count > 0:
    results = collection.get(include=["documents", "metadatas"])
    ids = results["ids"]
    docs = results["documents"]
    metas = results["metadatas"]

    # 统计来源文件
    sources = {}
    for meta in metas:
        src = meta.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    print("=== Source Files ===")
    for src, cnt in sources.items():
        print(f"  [{cnt} chunks] {src}")

    print()
    print("=== All Chunks ===")
    for i in range(len(ids)):
        doc = docs[i] or ""
        meta = metas[i] or {}
        src = meta.get("source", "unknown")
        filename = src.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
        print()
        print(f"-- Chunk {i+1} | File: {filename} --")
        preview = doc[:500] + "..." if len(doc) > 500 else doc
        print(preview)
else:
    print("Vector DB is empty.")
