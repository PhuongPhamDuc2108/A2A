import os
import json
import requests
import asyncio
from typing import List, Optional, AsyncGenerator
from queue import Queue
from threading import Thread

from pydantic import Field
from langchain_core.tools import tool, BaseTool
from langgraph.prebuilt import create_react_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, ToolCall
from tavily import TavilyClient

# SỬA LỖI: Đã xóa 'ContentType' và cập nhật import
from a2a.types import Part, TextPart


@tool
def search_tool(query: str) -> str:
    """Sử dụng công cụ này để tìm kiếm thông tin mới nhất trên internet."""
    try:
        tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        results = tavily_client.search(query, search_depth="basic", max_results=3)
        return json.dumps(results['results'])
    except Exception as e:
        return f"Lỗi khi tìm kiếm: {e}"


class GPTOSSChatModel(BaseChatModel):
    base_url: str = Field(...)
    model: str = Field(...)
    _bound_tools: Optional[List[BaseTool]] = None

    def bind_tools(self, tools: List[BaseTool], **kwargs) -> "GPTOSSChatModel":
        clone = self.model_copy(deep=True)
        clone._bound_tools = tools
        return clone

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        api_messages = []
        for m in messages:
            role = "user" if m.type == "human" else "assistant" if m.type == "ai" else "tool"
            content = m.content if isinstance(m.content, str) else json.dumps(m.content)
            api_messages.append({"role": role, "content": content})

        api_tools = []
        if self._bound_tools:
            for t in self._bound_tools:
                schema = t.args_schema.schema() if t.args_schema else {"type": "object", "properties": {}}
                api_tools.append({
                    "type": "function",
                    "function": {"name": t.name, "description": t.description, "parameters": schema}
                })

        payload = {
            "messages": api_messages, "temperature": 0.7, "top_p": 0.8, "top_k": 20,
            "presence_penalty": 1.5, "chat_template_kwargs": {"enable_thinking": True}, "tools": api_tools
        }
        headers = {"Content-Type": "application/json"}

        resp = requests.post(self.base_url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        tool_calls, content = [], ""
        if "response" in data:
            tool_calls, content = data["response"].get("tool_calls", []), data["response"].get("content", "")
        elif "choices" in data:
            choice0 = data["choices"][0]["message"]
            content, tool_calls = choice0.get("content", ""), choice0.get("tool_calls", [])

        if not tool_calls:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content or ""))])

        lc_tool_calls = [
            ToolCall(
                name=tc["function"]["name"],
                args=json.loads(tc["function"]["arguments"]),
                id=tc.get("id") or f"call_{hash(json.dumps(tc))}"
            ) for tc in tool_calls
        ]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="", tool_calls=lc_tool_calls))])

    @property
    def _llm_type(self) -> str:
        return "gpt-oss-custom-search"


class SearchAgent:
    # SỬA LỖI: Cập nhật cách khai báo content types
    SUPPORTED_CONTENT_TYPES = [Part(root=TextPart(text=""))]

    def __init__(self, llm: BaseChatModel, tools: List[BaseTool]):
        self.agent_executor = create_react_agent(llm, tools)

    async def stream(self, query: str, context_id: str) -> AsyncGenerator[dict, None]:
        q = Queue()

        def _run_in_thread():
            try:
                inputs = {"messages": [HumanMessage(content=query)]}
                for chunk in self.agent_executor.stream(inputs, stream_mode="values"):
                    q.put(chunk)
                q.put(None)
            except Exception as e:
                q.put(e)

        Thread(target=_run_in_thread, daemon=True).start()

        final_answer = ""
        while True:
            chunk = await asyncio.to_thread(q.get)
            if chunk is None: break
            if isinstance(chunk, Exception): raise chunk

            last_message = chunk['messages'][-1]

            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                yield {'is_task_complete': False, 'content': "Đang tìm kiếm trên internet..."}
            elif isinstance(last_message, ToolMessage):
                yield {'is_task_complete': False, 'content': "Đang xử lý thông tin tìm được..."}
            elif isinstance(last_message, AIMessage) and not last_message.tool_calls:
                final_answer = last_message.content
                break

        yield {'is_task_complete': True, 'content': final_answer}

