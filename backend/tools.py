from typing import Optional, List
import os
import requests
import datetime
import threading
import json
from dotenv import load_dotenv
try:
    from langchain_core.tools import tool
except ImportError:
    from langchain_core.tools import tool

load_dotenv()

AMAP_WEATHER_API = os.getenv("AMAP_WEATHER_API")
AMAP_API_KEY = os.getenv("AMAP_API_KEY")

_LAST_RAG_CONTEXT = None
_KNOWLEDGE_TOOL_CALLS_THIS_TURN = 0
_RAG_STEP_QUEUE = None  # asyncio.Queue, set by agent before streaming
_RAG_STEP_LOOP = None   # asyncio loop, captured when setting queue


def _set_last_rag_context(context: dict):
    global _LAST_RAG_CONTEXT
    _LAST_RAG_CONTEXT = context


def get_last_rag_context(clear: bool = True) -> Optional[dict]:
    """获取最近一次 RAG 检索上下文，默认读取后清空。"""
    global _LAST_RAG_CONTEXT
    context = _LAST_RAG_CONTEXT
    if clear:
        _LAST_RAG_CONTEXT = None
    return context


def reset_tool_call_guards():
    """每轮对话开始时重置工具调用计数。"""
    global _KNOWLEDGE_TOOL_CALLS_THIS_TURN
    _KNOWLEDGE_TOOL_CALLS_THIS_TURN = 0


def set_rag_step_queue(queue):
    """设置 RAG 步骤队列，并捕获当前事件循环以便跨线程调度。"""
    global _RAG_STEP_QUEUE, _RAG_STEP_LOOP
    _RAG_STEP_QUEUE = queue
    if queue:
        import asyncio
        try:
            _RAG_STEP_LOOP = asyncio.get_running_loop()
        except RuntimeError:
            _RAG_STEP_LOOP = asyncio.get_event_loop()
    else:
        _RAG_STEP_LOOP = None


def emit_rag_step(icon: str, label: str, detail: str = ""):
    """向队列发送一个 RAG 检索步骤。支持跨线程安全调用。"""
    global _RAG_STEP_QUEUE, _RAG_STEP_LOOP
    if _RAG_STEP_QUEUE is not None and _RAG_STEP_LOOP is not None:
        step = {"icon": icon, "label": label, "detail": detail}
        try:
            if not _RAG_STEP_LOOP.is_closed():
                _RAG_STEP_LOOP.call_soon_threadsafe(_RAG_STEP_QUEUE.put_nowait, step)
        except Exception:
            pass

reminders = []

# 提醒工具函数
def add_reminder(reminder_info):
    """
    添加提醒事项
    格式："提醒内容,提醒时间"，例如："吃药,2026-03-03 12:00"
    """
    try:
        content, time_str = reminder_info.split(',', 1)
        reminder_time = datetime.datetime.strptime(time_str.strip(), "%Y-%m-%d %H:%M")
        reminders.append({"content": content.strip(), "time": reminder_time})
        # 启动一个线程来检查提醒
        def check_reminder():
            while True:
                now = datetime.datetime.now()
                print(f"当前系统时间：{now.strftime('%Y-%m-%d %H:%M:%S')}")  # 打印当前时间
                for reminder in reminders[:]:
                    if now >= reminder["time"]:
                        print(f"\n🔔 提醒：{reminder['content']}")
                        reminders.remove(reminder)
                import time
                time.sleep(60)  # 每分钟检查一次
        
        # 只启动一次检查线程
        if not hasattr(add_reminder, "thread_started"):
            thread = threading.Thread(target=check_reminder, daemon=True)
            thread.start()
            add_reminder.thread_started = True
        
        return f"已添加提醒：{content.strip()}，时间：{time_str.strip()}\n当前系统时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    except Exception as e:
        return f"添加提醒失败：{str(e)}，请使用正确格式：提醒内容,提醒时间（例如：吃药,2026-03-03 12:00）"


def get_current_weather(location: str, extensions: Optional[str] = "base") -> str:
    """获取天气信息"""
    if not location:
        return "location参数不能为空"
    if extensions not in ("base", "all"):
        return "extensions参数错误，请输入base或all"

    if not AMAP_WEATHER_API or not AMAP_API_KEY:
        return "天气服务未配置（缺少 AMAP_WEATHER_API 或 AMAP_API_KEY）"

    params = {
        "key": AMAP_API_KEY,
        "city": location,
        "extensions": extensions,
        "output": "json",
    }

    try:
        resp = requests.get(AMAP_WEATHER_API, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "1":
            return f"查询失败：{data.get('info', '未知错误')}"

        if extensions == "base":
            lives = data.get("lives", [])
            if not lives:
                return f"未查询到 {location} 的天气数据"
            w = lives[0]
            return (
                f"【{w.get('city', location)} 实时天气】\n"
                f"天气状况：{w.get('weather', '未知')}\n"
                f"温度：{w.get('temperature', '未知')}℃\n"
                f"湿度：{w.get('humidity', '未知')}%\n"
                f"风向：{w.get('winddirection', '未知')}\n"
                f"风力：{w.get('windpower', '未知')}级\n"
                f"更新时间：{w.get('reporttime', '未知')}"
            )

        forecasts = data.get("forecasts", [])
        if not forecasts:
            return f"未查询到 {location} 的天气预报数据"
        f0 = forecasts[0]
        out = [f"【{f0.get('city', location)} 天气预报】", f"更新时间：{f0.get('reporttime', '未知')}", ""]
        today = (f0.get("casts") or [])[0] if f0.get("casts") else {}
        out += [
            "今日天气：",
            f"  白天：{today.get('dayweather','未知')}",
            f"  夜间：{today.get('nightweather','未知')}",
            f"  气温：{today.get('nighttemp','未知')}~{today.get('daytemp','未知')}℃",
        ]
        return "\n".join(out)

    except requests.exceptions.Timeout:
        return "错误：请求天气服务超时"
    except requests.exceptions.RequestException as e:
        return f"错误：天气服务请求失败 - {e}"
    except Exception as e:
        return f"错误：解析天气数据失败 - {e}"


@tool("search_knowledge_base")
def search_knowledge_base(query: str) -> str:
    """Search for information in the knowledge base using hybrid retrieval (dense + sparse vectors)."""
    # ... guards omitted ...
    global _KNOWLEDGE_TOOL_CALLS_THIS_TURN
    if _KNOWLEDGE_TOOL_CALLS_THIS_TURN >= 1:
        return (
            "TOOL_CALL_LIMIT_REACHED: search_knowledge_base has already been called once in this turn. "
            "Use the existing retrieval result and provide the final answer directly."
        )
    _KNOWLEDGE_TOOL_CALLS_THIS_TURN += 1

    from .rag_pipeline import run_rag_graph

    # 在同步工具中获取当前的 Loop 可能不可靠，但我们之前是通过 call_soon_threadsafe 调度的。
    # 这里 _RAG_STEP_QUEUE 是在主线程/Loop 设置的全局变量。
    # 如果工具运行在线程池中，它是可以访问到全局变量 _RAG_STEP_QUEUE 的。
    # emit_rag_step 内部做了 try-except 和 get_event_loop()。

    # 问题可能出在 asyncio.get_event_loop() 在子线程中调用会报错或者拿不到主线程的loop。
    # 我们应该在 set_rag_step_queue 时也保存 loop 引用，或者在 emit_rag_step 中更健壮地获取 loop。

    rag_result = run_rag_graph(query)

    docs = rag_result.get("docs", []) if isinstance(rag_result, dict) else []
    rag_trace = rag_result.get("rag_trace", {}) if isinstance(rag_result, dict) else {}
    if rag_trace:
        _set_last_rag_context({"rag_trace": rag_trace})

    if not docs:
        return "No relevant documents found in the knowledge base."

    formatted = []
    for i, result in enumerate(docs, 1):
        source = result.get("filename", "Unknown")
        page = result.get("page_number", "N/A")
        text = result.get("text", "")
        formatted.append(f"[{i}] {source} (Page {page}):\n{text}")

    return "Retrieved Chunks:\n" + "\n\n---\n\n".join(formatted)


@tool("list_directory")
def list_directory(path: str = ".") -> str:
    """列出指定目录下的文件和文件夹。
    
    Args:
        path: 目录路径，默认为当前目录（.）
        
    Returns:
        目录内容的格式化列表
    """
    try:
        # 确保路径是绝对路径
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        
        # 安全检查：防止访问敏感目录
        sensitive_dirs = [
            os.path.expanduser("~"),  # 用户目录
            "C:\\Windows", "C:\\Program Files", "C:\\Program Files (x86)",
            "C:\\Users", "C:\\ProgramData"
        ]
        
        for sensitive_dir in sensitive_dirs:
            if path.lower().startswith(sensitive_dir.lower()):
                return f"安全限制：禁止访问敏感目录 {sensitive_dir}"
        
        # 检查目录是否存在
        if not os.path.exists(path):
            return f"错误：目录 '{path}' 不存在"
        
        if not os.path.isdir(path):
            return f"错误：'{path}' 不是一个目录"
        
        # 获取目录内容
        items = []
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                items.append({"name": item, "type": "directory", "path": item_path})
            else:
                items.append({"name": item, "type": "file", "path": item_path})
        
        # 格式化输出
        result = f"目录内容 ({path}):\n\n"
        for item in items:
            icon = "📁" if item["type"] == "directory" else "📄"
            result += f"{icon} {item['name']}\n"
        
        return result
        
    except PermissionError:
        return f"权限错误：无法访问目录 '{path}'"
    except Exception as e:
        return f"错误：{str(e)}"


@tool("read_file")
def read_file(file_path: str, encoding: str = "utf-8") -> str:
    """读取指定文件的内容。
    
    Args:
        file_path: 文件路径
        encoding: 文件编码，默认为utf-8
        
    Returns:
        文件内容或错误信息
    """
    try:
        # 确保路径是绝对路径
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)
        
        # 安全检查：防止访问敏感目录
        sensitive_dirs = [
            os.path.expanduser("~"),
            "C:\\Windows", "C:\\Program Files", "C:\\Program Files (x86)",
            "C:\\Users", "C:\\ProgramData"
        ]
        
        for sensitive_dir in sensitive_dirs:
            if file_path.lower().startswith(sensitive_dir.lower()):
                return f"安全限制：禁止访问敏感目录中的文件"
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return f"错误：文件 '{file_path}' 不存在"
        
        if not os.path.isfile(file_path):
            return f"错误：'{file_path}' 不是一个文件"
        
        # 检查文件大小（限制为10MB）
        file_size = os.path.getsize(file_path)
        max_size = 10 * 1024 * 1024  # 10MB
        if file_size > max_size:
            return f"错误：文件过大（{file_size / 1024 / 1024:.2f}MB），最大支持10MB"
        
        # 读取文件内容
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            content = f.read()
        
        # 限制返回内容大小（最多10000字符）
        max_chars = 10000
        if len(content) > max_chars:
            content = content[:max_chars] + "\n\n[文件内容过长，已截断]"
        
        return f"文件内容 ({file_path}):\n\n{content}"
        
    except PermissionError:
        return f"权限错误：无法读取文件 '{file_path}'"
    except UnicodeDecodeError:
        return f"编码错误：无法使用 {encoding} 编码读取文件，尝试其他编码"
    except Exception as e:
        return f"错误：{str(e)}"
