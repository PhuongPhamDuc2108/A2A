import os
import json
import requests
import asyncio
import logging
from typing import List, Optional, AsyncGenerator, Dict, Any
from queue import Queue
from threading import Thread

import httpx
from pydantic import Field, BaseModel
from langchain_core.tools import Tool, BaseTool
from langgraph.prebuilt import create_react_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from a2a.client import A2AClient
# THAY ĐỔI: Đã xóa 'ContentType' khỏi dòng import
from a2a.types import Task, Message, Part, TextPart

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Lớp GPTOSSChatModel (giữ nguyên để giao tiếp với LLM của bạn)
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
            role = "user" if m.type == "human" else "assistant" if m.type == "ai" else "tool" if m.type == "tool" else "system"
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
            "messages": api_messages, "temperature": 0.1, "top_p": 0.8, "top_k": 20,
            "presence_penalty": 1.5, "chat_template_kwargs": {"enable_thinking": True}, "tools": api_tools
        }
        headers = {"Content-Type": "application/json"}

        print("\n=== Sending payload from HOST AGENT to GPT-OSS ===")
        print(json.dumps(payload, ensure_ascii=False, indent=2))

        resp = requests.post(self.base_url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        print("\n=== GPT-OSS Response to HOST AGENT ===")
        print(json.dumps(data, ensure_ascii=False, indent=2))

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
        return "gpt-oss-custom-host"


class HostAgent:
    """
    Host Agent đóng vai trò điều phối, gọi đến các remote agent khác để thực hiện tác vụ.
    """
    # THAY ĐỔI: Cập nhật cách khai báo content types
    SUPPORTED_CONTENT_TYPES = [Part(root=TextPart(text=""))]

    SYSTEM_PROMPT_TEMPLATE = """You are an expert delegator that can delegate the user request to the
appropriate remote agents.

Discovery:
- You can use `list_remote_agents` to list the available remote agents you
can use to delegate the task.

Execution:
- For actionable requests, you can use `send_message_to_agent` to interact with remote agents to take action.

Be sure to include the remote agent name when you respond to the user.

Please rely on tools to address the request, and don't make up the response. If you are not sure, please ask the user for more details.
Focus on the most recent parts of the conversation primarily.

Agents:
{agents_list}
"""

    def __init__(self):
        # 1. Khám phá các remote agent và lưu thông tin
        self.remote_agents: List[Dict[str, Any]] = asyncio.run(self._discover_remote_agents())
        if not self.remote_agents:
            raise RuntimeError("Could not discover any remote agents. Check URLs in .env")

        # 2. Tạo các tool
        self.tools = self._create_tools()

        # 3. Tạo system prompt động
        system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
            agents_list=json.dumps(self.remote_agents, indent=2)
        )

        # 4. Khởi tạo LLM và Agent Executor
        self.llm = GPTOSSChatModel(
            base_url=os.getenv("TOOL_LLM_URL"),
            model=os.getenv("TOOL_LLM_NAME"),
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("placeholder", "{messages}"),
        ])

        self.agent_executor = create_react_agent(self.llm, self.tools, messages_modifier=prompt)

    async def _discover_remote_agents(self) -> List[Dict[str, Any]]:
        """
        Đọc AgentCard từ các URL trong .env và trả về danh sách thông tin agent.
        """
        agent_urls = [url.strip() for url in os.getenv("REMOTE_AGENT_URLS", "").split(",") if url.strip()]
        discovered_agents = []
        async with httpx.AsyncClient() as client:
            tasks = [A2AClient(base_url=url, client=client).get_agent_card() for url in agent_urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for agent_card in results:
                if isinstance(agent_card, Exception):
                    logger.error(f"Failed to fetch agent card: {agent_card}")
                    continue

                # Lưu thông tin cần thiết để LLM có thể quyết định
                discovered_agents.append({
                    "name": agent_card.name.replace(" ", "_"),
                    "url": agent_card.url,
                    "description": agent_card.description,
                    "skills": [s.description for s in agent_card.skills],
                })
        logger.info(f"Discovered {len(discovered_agents)} remote agents.")
        return discovered_agents

    def _create_tools(self) -> List[Tool]:
        """Tạo các tool cố định cho Host Agent."""

        def list_remote_agents_func() -> str:
            """Lists available remote agents you can delegate tasks to."""
            return json.dumps(self.remote_agents, indent=2)

        async def send_message_to_agent_func(agent_name: str, query: str) -> str:
            """Sends a message to a specific remote agent to perform a task."""
            agent_info = next((agent for agent in self.remote_agents if agent['name'] == agent_name), None)
            if not agent_info:
                return f"Error: Agent '{agent_name}' not found. Use 'list_remote_agents' to see available agents."

            try:
                async with httpx.AsyncClient() as client:
                    a2a_client = A2AClient(base_url=agent_info['url'], client=client)
                    message = Message(role='user', content=[Part(root=TextPart(text=query))])
                    task_result = await a2a_client.execute(message=message, stream=False)

                    if task_result and isinstance(task_result, Task) and task_result.artifacts:
                        text_part = task_result.artifacts[-1].parts[0].root
                        if isinstance(text_part, TextPart):
                            return text_part.text
                    return "Agent did not return a valid text response."
            except Exception as e:
                logger.error(f"Error calling agent '{agent_name}': {e}")
                return f"An error occurred while communicating with the agent: {e}"

        tools = [
            Tool(name="list_remote_agents", func=list_remote_agents_func, description=list_remote_agents_func.__doc__),
            Tool(name="send_message_to_agent", func=send_message_to_agent_func,
                 description=send_message_to_agent_func.__doc__)
        ]
        return tools

    async def stream(self, query: str, context_id: str) -> AsyncGenerator[dict, None]:
        """Xử lý query của người dùng và stream các bước thực thi."""
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
                tool_call = last_message.tool_calls[0]
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                if tool_name == 'send_message_to_agent':
                    content = f"Contacting agent `{tool_args['agent_name']}` with query: `{tool_args['query']}`"
                else:
                    content = f"Using tool `{tool_name}`..."
                yield {'is_task_complete': False, 'content': content}

            elif isinstance(last_message, ToolMessage):
                yield {'is_task_complete': False, 'content': f"Received result from tool `{last_message.name}`."}

            elif isinstance(last_message, AIMessage) and not last_message.tool_calls:
                final_answer = last_message.content
                break

        yield {'is_task_complete': True, 'content': final_answer}

