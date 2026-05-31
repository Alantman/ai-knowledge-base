import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

# ===== 配置 =====
MAX=5

# ===== 模型 =====
model=ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)

# ===== 工具 =====
@tool
def add(a:int,b:int) ->int:
    """两个整数相加"""
    return a+b
@tool
def multiply(a:int,b:int) ->int:
    """两个整数相乘"""
    return a*b
Tools=[add,multiply]
TOOL_MAP={t.name:t for t in Tools}#等价于 TOOL_MAP={"add":add, "multiply":multiply} ->工具少的时候可以像这样写死
model_with_tools=model.bind_tools(Tools)

# ===== 手写 Agent 循环 =====
def user_input(user_input:str):
    messages=[HumanMessage(content=user_input)]
    iteration=0
    while iteration<MAX:
        iteration+=1
        ai_msg=model_with_tools.invoke(messages)
        #模型不再调用工具
        if not ai_msg.tool_calls:
            messages.append(ai_msg)
            break

        #模型继续调用工具，先将上一步的结果加到下一步的信息中
        messages.append(ai_msg)
        for tc in ai_msg.tool_calls:
            tool_fn=TOOL_MAP.get(tc["name"])
            if tool_fn:
                result=tool_fn.invoke(tc["args"])
                messages.append(
                    ToolMessage(content=str(result),tool_call_id=tc["id"])
                )

    return messages[-1].content

if __name__ == '__main__':
    print(user_input("先算5加6的结果，再把结果乘以7"))










