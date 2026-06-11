import os
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from config.ai_conf import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from services.doc_service import get_retriever

model = ChatOpenAI(
    base_url=DEEPSEEK_BASE_URL,
    api_key=DEEPSEEK_API_KEY,
    model=DEEPSEEK_MODEL,
    temperature=0.7,
    streaming=True,
)

# RAG 系统提示词：要求模型基于资料回答，并在末尾注明来源
RAG_SYSTEM_PROMPT = (
    "根据以下资料回答用户问题。"
    "如果资料中找不到相关信息，请直接告诉用户知识库中未找到相关信息，不要猜测或编造。"
    "回答时请在末尾用 [来源: 文件名] 格式注明信息来源。\n\n"
    "资料：\n{context}"
)

# 查询重写提示词：把含指代词的口语问题改写为独立检索查询
QUERY_REWRITE_PROMPT = (
    "将用户问题改写为一个适合文档检索的独立查询。"
    "去除指代词（它、那个、这个、他），结合对话历史补充上下文。"
    "直接输出改写后的查询文本，不要加任何解释。\n\n"
    "对话历史：\n{history}\n\n"
    "用户问题：{question}\n\n"
    "改写查询："
)

# 检索意图判断提示词：区分知识查询和闲聊/问候/自我介绍
NEED_RETRIEVAL_PROMPT = (
    "判断用户问题是否需要查询知识库文档来回答。"
    "需要检索：询问文档内容、专业知识、事实性信息、数据查询等问题。"
    "不需要检索：闲聊、自我介绍、问候、感谢、道别、简单确认等社交性对话。"
    "只回复'需要'或'不需要'，不要加任何解释。\n\n"
    "问题：{question}\n\n回答："
)

# 通用对话提示词（不需要检索时使用）
GENERAL_SYSTEM_PROMPT = "你是一个友好的AI助手，可以回答各种问题。"

# Agent 系统提示词：要求模型基于工具返回的资料回答，并注明来源
AGENT_SYSTEM_PROMPT = (
    "你是一个知识库问答助手。你可以使用工具搜索知识库中的文档来回答问题。"
    "当使用搜索工具获取资料后，请在回答末尾用 [来源: 文件名] 格式注明信息来源。"
    "如果知识库中找不到相关信息，请直接告诉用户，不要猜测或编造。"
)

# Memory 存储，按 session 隔离
store: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def _format_docs(docs: list) -> str:
    """格式化检索到的文档片段，带上来源文件名"""
    parts = []
    for i, doc in enumerate(docs):
        src = doc.metadata.get("source", "unknown")
        filename = os.path.basename(src) if src else "unknown"
        parts.append(f"[来源{i + 1}: {filename}]\n{doc.page_content}")
    return "\n\n".join(parts)


def _rewrite_query(question: str, history_messages: list) -> str:
    """用 LLM 改写查询：补全指代、去掉口语化，提升检索命中率"""
    if not history_messages:
        return question

    # 取最近 3 轮对话作为上下文
    history_text = "\n".join(
        f"{'用户' if isinstance(m, HumanMessage) else 'AI'}: {m.content[:200]}"
        for m in history_messages[-6:]
    )

    try:
        response = model.invoke(
            QUERY_REWRITE_PROMPT.format(history=history_text, question=question)
        )
        rewritten = response.content.strip()
        if rewritten and len(rewritten) > 1:
            return rewritten
    except Exception:
        pass

    return question


def ask_stream(question: str, session_id: str = "default"):
    """RAG 流式问答：查询重写 → 检索意图判断 → 按需检索 → 流式输出"""
    history = get_session_history(session_id)
    history_messages = list(history.messages)

    # 1. 查询重写
    rewritten = _rewrite_query(question, history_messages)
    if rewritten != question:
        print(f"[查询重写] {question[:60]}  →  {rewritten[:80]}")

    # 2. 检索意图判断：闲聊/问候/自我介绍 → 跳过检索
    need_retrieval = True
    try:
        check_result = model.invoke(NEED_RETRIEVAL_PROMPT.format(question=rewritten))
        need_retrieval = "不需要" not in check_result.content
        print(f"[检索意图] {rewritten[:60]}... → {'需要检索' if need_retrieval else '无需检索'}")
    except Exception:
        pass  # 判断失败默认走检索

    # 3. 检索（按需）
    if need_retrieval:
        retriever = get_retriever()
        docs = retriever.invoke(rewritten)
        context = _format_docs(docs)
        print(f"[检索] 命中 {len(docs)} 个片段 (k=6, MMR):")
        for i, doc in enumerate(docs):
            src = doc.metadata.get("source", "?")
            preview = doc.page_content[:80].replace("\n", " ")
            print(f"  [{i+1}] {os.path.basename(src)} → {preview}...")
        system_msg = SystemMessage(content=RAG_SYSTEM_PROMPT.format(context=context))
    else:
        system_msg = SystemMessage(content=GENERAL_SYSTEM_PROMPT)

    # 4. 拼接消息
    messages = [system_msg] + history_messages + [HumanMessage(content=question)]

    # 5. 流式输出
    full_answer = ""
    for chunk in model.stream(messages):
        if chunk.content:
            full_answer += chunk.content
            yield chunk.content

    # 5. 存入历史
    history.add_message(HumanMessage(content=question))
    history.add_message(AIMessage(content=full_answer))


# ============================================================
# Tool Calling（单轮）：模型自主决定是否搜索知识库
# ============================================================

from langchain_core.messages import ToolMessage
from services.tools import search_knowledge_base, list_knowledge_base_documents

# 工具名 → 函数 映射表，Agent 循环用
TOOL_MAP = {
    "search_knowledge_base": search_knowledge_base,
    "list_knowledge_base_documents": list_knowledge_base_documents,
}
tools=[search_knowledge_base, list_knowledge_base_documents]
model_with_tools = model.bind_tools(tools)   


def _get_agent_history(session_id: str) -> list:
    """Agent 模式复用 RAG 的记忆 store，不再单独维护"""
    hist = get_session_history(session_id)
    return list(hist.messages)


def ask_agent_stream(question: str, session_id: str = "default"):
    """流式问答，单轮 Tool Calling：模型决定是否调工具，最多一轮"""
    history_messages = _get_agent_history(session_id)
    current_messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)] + history_messages + [HumanMessage(content=question)]

    ai_msg = model_with_tools.invoke(current_messages)

    session_hist = get_session_history(session_id)

    if ai_msg.tool_calls:
        tool_msgs = []
        for tc in ai_msg.tool_calls:
            tool_fn = TOOL_MAP.get(tc["name"])
            if tool_fn:
                result = tool_fn.invoke(tc["args"])
                tool_msgs.append(
                    ToolMessage(content=result, tool_call_id=tc["id"])
                )

        current_messages.append(ai_msg)
        current_messages.extend(tool_msgs)

        full_answer = ""
        for chunk in model_with_tools.stream(current_messages):
            if chunk.content:
                full_answer += chunk.content
                yield chunk.content

        session_hist.add_message(HumanMessage(content=question))
        session_hist.add_message(AIMessage(content=full_answer))
    else:
        session_hist.add_message(HumanMessage(content=question))
        session_hist.add_message(ai_msg)
        yield ai_msg.content


# ============================================================
# Agent（多轮推理）：while 循环 → 模型可以连续调用多次工具
# ============================================================

MAX_ITERATIONS = 5  # 安全阀：最多走 5 轮，防止无限循环


def ask_agent_reasoning_stream(question: str, session_id: str = "default"):
    """Agent 多轮推理的流式版本 — 工具调用阶段静默，最终答案流式输出"""
    history_messages = _get_agent_history(session_id)
    messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)] + history_messages + [HumanMessage(content=question)]

    iteration = 0
    while iteration < MAX_ITERATIONS:
        iteration += 1

        ai_msg = model_with_tools.invoke(messages)

        if not ai_msg.tool_calls:
            break

        messages.append(ai_msg)
        for tc in ai_msg.tool_calls:
            tool_fn = TOOL_MAP.get(tc["name"])
            if tool_fn:
                result = tool_fn.invoke(tc["args"])
                messages.append(
                    ToolMessage(content=result, tool_call_id=tc["id"])
                )

    full_answer = ""
    for chunk in model_with_tools.stream(messages):
        if chunk.content:
            full_answer += chunk.content
            yield chunk.content

    session_hist = get_session_history(session_id)
    session_hist.add_message(HumanMessage(content=question))
    session_hist.add_message(AIMessage(content=full_answer))


# ============================================================
# Agent（LangGraph）：用图结构替代手动 while 循环
# ============================================================

from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AIMessageChunk

# 用 MessagesState 管理消息列表 — LangGraph 自动帮你追加消息到 state["messages"]
# 等价于你之前手动做的 messages.append()  然后传给下一轮

# ToolNode 等价于之前手动写的：
#   for tc in ai_msg.tool_calls:
#       result = search_knowledge_base.invoke(tc["args"])
#       messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

# tools_condition 等价于之前手动写的：
#   if ai_msg.tool_calls: → 走工具 / else: → END


def _build_langgraph_agent(): 
    """构建 LangGraph Agent，用图替代 while 循环"""

    def call_model(state: MessagesState):
        """agent 节点：调用 LLM。state["messages"] 是历史消息列表"""
        response = model_with_tools.invoke(state["messages"])
        # LangGraph 的 MessagesState 会用 {messages: [...]} 接收返回值，
        # 自动把 AIMessage 追加到消息列表里
        return {"messages": response}

    # 两个节点：agent（调 LLM） + tools（执行工具）
    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)   #调用 LLM
    graph.add_node("tools", ToolNode([search_knowledge_base, list_knowledge_base_documents]))  # 执行工具

    # 边：
    #   START → agent → (conditional) → tools → agent → ... → END
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile()


langgraph_agent = _build_langgraph_agent()


def ask_agent_langgraph_stream(question: str, session_id: str = "default"):
    """LangGraph Agent 流式推理：图结构自动管理循环和状态，不需要手动 while
    stream_mode="messages" token级流式输出
    """
    history_messages = _get_agent_history(session_id)

    full_answer = ""
    for chunk, metadata in langgraph_agent.stream(
        {"messages": [SystemMessage(content=AGENT_SYSTEM_PROMPT)] + history_messages + [HumanMessage(content=question)]},
        config={"recursion_limit": 5},
        stream_mode="messages",
    ):
        # chunk 是 AIMessageChunk — 和手动流式一样的逐 token 输出
        # 过滤掉 ToolMessage 和没有内容的 chunk
        if isinstance(chunk, AIMessageChunk) and chunk.content:
            full_answer += chunk.content
            yield chunk.content

        # 如果是 tool_calls chunk（工具调用阶段的流式片段），静默跳过
        # 工具执行发生在 tools 节点，不是流式输出

    session_hist = get_session_history(session_id)
    session_hist.add_message(HumanMessage(content=question))
    session_hist.add_message(AIMessage(content=full_answer))
