"""
一键启动脚本 — 同时启动后端(8010)和前端(5173)
"""
import subprocess
import os
import sys
import time
import webbrowser
import signal

ROOT = os.path.dirname(os.path.abspath(__file__))

print("=" * 50)
print("  AI 知识库问答系统 — 启动中...")
print("=" * 50)

# 启动后端
backend = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "main:app", "--host", "localhost", "--port", "8010"],
    cwd=os.path.join(ROOT, "backend"),
)

# 启动前端
frontend = subprocess.Popen(
    ["npm", "run", "dev"],
    cwd=os.path.join(ROOT, "frontend"),
    shell=True,
)

print()
print("  后端: http://localhost:8010")
print("  前端: http://localhost:5173")
print()
print("  在此窗口按 回车 停止所有服务")
print("=" * 50)

time.sleep(3)
webbrowser.open("http://localhost:5173")


def kill_process(proc):
    """Windows 上强制杀进程（含子进程）"""
    try:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True)
    except Exception:
        proc.kill()


input()
print("正在关闭...")
kill_process(backend)
kill_process(frontend)
print("已停止")
