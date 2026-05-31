"""LangGraph 三板斧 —— StateGraph + add_node + add_edge 替代手写 while"""
import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition


# ===== 配置 =====
MAX_ITERATIONS = 10

# ===== 模型 =====
model = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)


# ===== 工具 =====
@tool
def add(a: int, b: int) -> int:
    """两个整数相加"""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """两个整数相乘"""
    return a * b


tools = [add, multiply]
model_with_tools = model.bind_tools(tools)


# ===== LangGraph Agent =====
def call_model(state: MessagesState):
    """Agent 节点：调用模型"""
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# 三板斧
graph = StateGraph(MessagesState)
graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")

app = graph.compile()


if __name__ == "__main__":
    result = app.invoke({"messages": [("user", "先算 3 加 5 等于多少，再把结果乘以 2")]})
    print(result["messages"][-1].content)
