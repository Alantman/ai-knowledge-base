"""
测试运行脚本
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

passed = 0
failed = 0


def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}")


# ---- 1. _format_docs ----
from services.rag_service import _format_docs

docs = [
    Document(page_content="A", metadata={"source": "/path/doc1.pdf"}),
    Document(page_content="B", metadata={"source": "/path/doc2.txt"}),
]
r = _format_docs(docs)
check("format_docs 含来源和文件名", "[来源1: doc1.pdf]" in r and "[来源2: doc2.txt]" in r)
check("format_docs 无来源显示unknown", "[来源1: unknown]" in _format_docs([Document(page_content="x", metadata={})]))
check("format_docs 空列表", _format_docs([]) == "")

# ---- 2. session 隔离 ----
from services.rag_service import store, get_session_history
store.clear()
a = get_session_history("a")
b = get_session_history("b")
a.add_message(HumanMessage(content="A的问题"))
b.add_message(HumanMessage(content="B的问题"))
check("session A 独立", len(a.messages) == 1 and a.messages[0].content == "A的问题")
check("session B 独立", len(b.messages) == 1 and b.messages[0].content == "B的问题")
check("同一 session 复用", get_session_history("a") is a)
store.clear()

# ---- 3. 查询重写 ----
from services.rag_service import _rewrite_query
check("无历史不调LLM", _rewrite_query("它怎么用", []) == "它怎么用")

from unittest.mock import patch, MagicMock
with patch("services.rag_service.model") as mock_llm:
    mock_llm.invoke.return_value.content = "Python list 排序"
    r = _rewrite_query("它怎么排序", [HumanMessage(content="list"), AIMessage(content="append")])
    check("有历史时改写", r == "Python list 排序")

with patch("services.rag_service.model") as mock_llm:
    mock_llm.invoke.side_effect = Exception("error")
    r = _rewrite_query("它怎么用", [HumanMessage(content="hello")])
    check("LLM失败fallback", r == "它怎么用")

# ---- 4. Prompt 内容 ----
from services.rag_service import RAG_SYSTEM_PROMPT, NEED_RETRIEVAL_PROMPT
check("RAG提示词含来源", "[来源" in RAG_SYSTEM_PROMPT)
check("RAG提示词含不要编造", "不要猜测或编造" in RAG_SYSTEM_PROMPT)
check("检索意图提示词含需要/不需要", "需要" in NEED_RETRIEVAL_PROMPT and "不需要" in NEED_RETRIEVAL_PROMPT)

# ---- 5. 闲聊跳过检索 ----
from services.rag_service import ask_stream
with patch("services.rag_service.model") as mock_llm, \
     patch("services.rag_service.get_retriever") as mock_ret:
    store.clear()
    mock_llm.invoke.return_value.content = "不需要"
    mock_llm.stream.return_value = iter([MagicMock(content="你好")])
    list(ask_stream("你好", session_id="t"))
    check("闲聊不调检索", not mock_ret.called)
    store.clear()

# ---- 6. 工具搜索 ----
from services.tools import search_knowledge_base
with patch("services.tools.get_retriever") as mock_ret:
    mock_ret.return_value.invoke.return_value = [
        Document(page_content="test", metadata={"source": "/data/test.pdf"})
    ]
    r = search_knowledge_base.invoke({"query": "x"})
    check("工具结果带来源", "[来源1: test.pdf]" in r)

    mock_ret.return_value.invoke.return_value = []
    r = search_knowledge_base.invoke({"query": "x"})
    check("无结果提示", "未找到" in r)


print(f"\n{'='*40}")
print(f"  {passed} passed, {failed} failed")
print(f"{'='*40}")
