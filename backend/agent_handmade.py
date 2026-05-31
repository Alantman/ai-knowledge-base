"""手写 Agent 循环 —— 自己管 while、append、break"""
import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

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
TOOL_MAP = {t.name: t for t in tools}
model_with_tools = model.bind_tools(tools)


# ===== 手写 Agent 循环 =====
def run_agent(user_input: str):
    messages = [HumanMessage(content=user_input)]
    iteration = 0

    while iteration < MAX_ITERATIONS:
        iteration += 1

        ai_msg = model_with_tools.invoke(messages)

        # 模型觉得不需要调工具，回答完了
        if not ai_msg.tool_calls:
            messages.append(ai_msg)
            break

        # 模型要调工具，执行然后结果塞回去
        messages.append(ai_msg)
        for tc in ai_msg.tool_calls:
            tool_fn = TOOL_MAP.get(tc["name"])
            if tool_fn:
                result = tool_fn.invoke(tc["args"])
                messages.append(
                    ToolMessage(content=str(result), tool_call_id=tc["id"])
                )

    return messages[-1].content


if __name__ == "__main__":
    print(run_agent("先算 3 加 5 等于多少，再把结果乘以 2"))
