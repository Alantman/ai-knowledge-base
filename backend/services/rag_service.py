from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from config.ai_conf import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from services.doc_service import get_retriever

model = ChatOpenAI(
    base_url=DEEPSEEK_BASE_URL,
    api_key=DEEPSEEK_API_KEY,
    model=DEEPSEEK_MODEL,
    temperature=0.7,
    streaming=True,
)

# RAG 提示词：retriever 查到的文档作为 context，用户问题作为 question
prompt = ChatPromptTemplate.from_messages([
    ("system", "根据以下资料回答用户问题。如果资料中找不到相关信息，可以根据自己的知识和对话历史来回答。\n\n资料：{context}"),
    MessagesPlaceholder(variable_name="history", optional=True),
    ("user", "{question}"),
])

# RAG Chain：检索 → 拼 prompt → 模型 → 解析
# 注意：必须把 history 也传下去，否则 RunnableWithMessageHistory 注入的
# 历史消息会被 dict 构造函数丢掉
rag_chain = (
    {
        "context": itemgetter("question") | get_retriever(),
        "question": itemgetter("question"),
        "history": itemgetter("history"),
    }
    | prompt
    | model
    | StrOutputParser()
)

# Memory 存储，按 session 隔离
store: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


chain_with_memory = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)


def ask_stream(question: str, session_id: str = "default"):
    """流式问答，逐 chunk yield。手动管理历史以解决 RunnableWithMessageHistory 的缓冲问题。"""
    from langchain_core.messages import HumanMessage, AIMessage

    history = get_session_history(session_id)
    history_messages = history.messages

    full_answer = ""
    for chunk in rag_chain.stream({
        "question": question,
        "history": history_messages,
    }):
        full_answer += chunk
        yield chunk

    # 存入历史（和 RunnableWithMessageHistory 一样的逻辑）
    history.add_message(HumanMessage(content=question))
    history.add_message(AIMessage(content=full_answer))


# ============================================================
# Tool Calling（单轮）：模型自主决定是否搜索知识库
# ============================================================

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from services.tools import search_knowledge_base

# 工具名 → 函数 映射表，Agent 循环用
TOOL_MAP = {
    "search_knowledge_base": search_knowledge_base,
}

model_with_tools = model.bind_tools([search_knowledge_base])


def _get_agent_history(session_id: str) -> list:
    """Agent 模式复用 RAG 的记忆 store，不再单独维护"""
    hist = get_session_history(session_id)
    return list(hist.messages)


def ask_agent_stream(question: str, session_id: str = "default"):
    """流式问答，单轮 Tool Calling：模型决定是否调工具，最多一轮"""
    history_messages = _get_agent_history(session_id)
    current_messages = history_messages + [HumanMessage(content=question)]

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
    messages = history_messages + [HumanMessage(content=question)]

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
    graph.add_node("tools", ToolNode([search_knowledge_base]))  # 执行工具

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
        {"messages": history_messages + [HumanMessage(content=question)]},
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


# ============================================================
# 临时文件分析：文件内容直接放入 prompt，不走 RAG
# ============================================================

file_prompt = ChatPromptTemplate.from_messages([
    ("system", "根据以下文件内容回答用户问题。如果文件内容不足以回答问题，可以结合你自己的知识补充。\n\n文件内容：\n{file_content}"),
    MessagesPlaceholder(variable_name="history", optional=True),
    ("user", "{question}"),
])

file_chain = (
    {
        "file_content": itemgetter("file_content"),
        "question": itemgetter("question"),
        "history": itemgetter("history"),
    }
    | file_prompt
    | model
    | StrOutputParser()
)


def ask_with_file_stream(file_content: str, question: str, session_id: str = "default"):
    """临时文件分析：直接把文件内容作为上下文，不走向量检索"""
    from langchain_core.messages import HumanMessage, AIMessage

    history = get_session_history(session_id)

    full_answer = ""
    for chunk in file_chain.stream({
        "file_content": file_content,
        "question": question,
        "history": history.messages,
    }):
        full_answer += chunk
        yield chunk

    history.add_message(HumanMessage(content=question))
    history.add_message(AIMessage(content=full_answer))
