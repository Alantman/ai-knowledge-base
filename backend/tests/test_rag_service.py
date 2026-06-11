"""
RAG 服务测试

运行方式：在项目根目录执行
  python -m pytest backend/tests/ -v
"""
import sys
import os

# 让测试能 import 项目代码
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage


# ============================================================
# 1. 纯函数测试：_format_docs — 不需要 Mock
# ============================================================

def test_format_docs_with_source():
    """格式化检索结果：应包含文件名和来源编号"""
    from services.rag_service import _format_docs

    docs = [
        Document(page_content="内容A", metadata={"source": "/path/to/doc1.pdf"}),
        Document(page_content="内容B", metadata={"source": "/path/to/doc2.txt"}),
    ]
    result = _format_docs(docs)

    assert "[来源1: doc1.pdf]" in result
    assert "[来源2: doc2.txt]" in result
    assert "内容A" in result
    assert "内容B" in result


def test_format_docs_unknown_source():
    """无来源信息时：显示 unknown"""
    from services.rag_service import _format_docs

    docs = [Document(page_content="无来源", metadata={})]
    result = _format_docs(docs)

    assert "[来源1: unknown]" in result


def test_format_docs_empty():
    """空列表返回空字符串"""
    from services.rag_service import _format_docs

    assert _format_docs([]) == ""


# ============================================================
# 2. Session 隔离测试
# ============================================================

def test_session_isolation():
    """不同 session_id 应有独立的对话历史"""
    from services.rag_service import store, get_session_history

    # 清空 store 避免之前测试干扰
    store.clear()

    hist_a = get_session_history("session_a")
    hist_b = get_session_history("session_b")

    hist_a.add_message(HumanMessage(content="A的问题"))
    hist_b.add_message(HumanMessage(content="B的问题"))

    # A 的历史里只有 A，B 的历史里只有 B
    assert len(hist_a.messages) == 1
    assert hist_a.messages[0].content == "A的问题"
    assert len(hist_b.messages) == 1
    assert hist_b.messages[0].content == "B的问题"

    # 再次获取同一 session 应该是同一个对象
    assert get_session_history("session_a") is hist_a

    store.clear()


# ============================================================
# 3. 查询重写测试（Mock LLM）
# ============================================================

def test_rewrite_query_no_history_returns_original():
    """无历史时直接返回原问题，不调 LLM"""
    from services.rag_service import _rewrite_query

    result = _rewrite_query("它怎么用", [])
    assert result == "它怎么用"


@patch("services.rag_service.model")
def test_rewrite_query_with_history(mock_model):
    """有历史时应调动 LLM 改写"""
    from services.rag_service import _rewrite_query

    mock_model.invoke.return_value.content = "Python list 排序方法"

    history = [
        HumanMessage(content="Python list 有哪些方法"),
        AIMessage(content="list 有很多方法如 append..."),
    ]
    result = _rewrite_query("它怎么排序", history)

    assert result == "Python list 排序方法"
    assert mock_model.invoke.called


@patch("services.rag_service.model")
def test_rewrite_query_llm_fails_fallback(mock_model):
    """LLM 失败时 fallback 返回原问题"""
    from services.rag_service import _rewrite_query

    mock_model.invoke.side_effect = Exception("API 挂了")

    history = [HumanMessage(content="Python"), AIMessage(content="嗯")]
    result = _rewrite_query("它怎么用", history)

    assert result == "它怎么用"  # fallback


# ============================================================
# 4. Prompt 内容测试 — 确保提示词要求正确
# ============================================================

def test_rag_prompt_requires_source_citation():
    """RAG 提示词应要求注明来源"""
    from services.rag_service import RAG_SYSTEM_PROMPT

    assert "[来源" in RAG_SYSTEM_PROMPT
    assert "不要猜测或编造" in RAG_SYSTEM_PROMPT
    assert "{context}" in RAG_SYSTEM_PROMPT


def test_retrieval_intent_prompt_outputs_only_need_or_not():
    """检索意图判断提示词应只回复'需要'或'不需要'"""
    from services.rag_service import NEED_RETRIEVAL_PROMPT

    assert "需要" in NEED_RETRIEVAL_PROMPT
    assert "不需要" in NEED_RETRIEVAL_PROMPT
    assert "不要加任何解释" in NEED_RETRIEVAL_PROMPT


# ============================================================
# 5. RAG 流式问答测试（Mock LLM + Retriever）
# ============================================================

@patch("services.rag_service.model")
@patch("services.rag_service.get_retriever")
def test_ask_stream_skips_retrieval_for_greeting(mock_retriever, mock_model):
    """闲聊/问候应跳过检索，模型回答后存入历史"""
    from services.rag_service import ask_stream, store, get_session_history

    store.clear()
    sid = "test_greeting"

    # Mock：意图判断返回"不需要"，流式回答返回"你好呀"
    mock_model.invoke.return_value.content = "不需要"
    mock_model.stream.return_value = iter([
        MagicMock(content="你好呀，"),
        MagicMock(content="有什么可以帮你的？"),
    ])

    chunks = list(ask_stream("你好", session_id=sid))

    # 验证：没有触发检索
    mock_retriever.assert_not_called()

    # 验证：流式内容正确
    assert "".join(chunks) == "你好呀，有什么可以帮你的？"

    # 验证：对话已存入历史
    hist = get_session_history(sid)
    assert len(hist.messages) == 2
    assert hist.messages[0].content == "你好"

    store.clear()


# ============================================================
# 6. 工具测试（需要 Mock Retriever）
# ============================================================

@patch("services.tools.get_retriever")
def test_search_knowledge_base_includes_source(mock_retriever):
    """搜索工具返回结果应包含来源标注"""
    from services.tools import search_knowledge_base

    mock_retriever.return_value.invoke.return_value = [
        Document(
            page_content="这是一段测试内容",
            metadata={"source": "/data/test.pdf"},
        )
    ]

    result = search_knowledge_base.invoke({"query": "测试"})

    assert "[来源1: test.pdf]" in result
    assert "测试内容" in result


@patch("services.tools.get_retriever")
def test_search_knowledge_base_empty_result(mock_retriever):
    """无结果时应返回提示"""
    from services.tools import search_knowledge_base

    mock_retriever.return_value.invoke.return_value = []

    result = search_knowledge_base.invoke({"query": "不存在"})

    assert "未找到" in result
