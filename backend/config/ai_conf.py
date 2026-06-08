import os
from pathlib import Path

# 从项目根目录的 .env 加载配置
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# DeepSeek 大模型
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# 硅基流动 Embedding（DeepSeek 不提供 Embedding）
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")

# Chroma 向量库路径
CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "chroma_db")

# 文档切分配置
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# 检索配置
RETRIEVER_K = 6
RETRIEVER_FETCH_K = 12  # MMR 初选候选数，需大于 k
