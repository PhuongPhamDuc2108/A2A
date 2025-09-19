import os
import json
import requests
import asyncio
import logging
from typing import List, Optional, AsyncGenerator, Dict, Any
from queue import Queue
from threading import Thread

import httpx
# SỬA LỖI: Import thêm BaseModel để định nghĩa schema
from pydantic import Field, BaseModel
from langchain_core.tools import Tool, BaseTool
from langgraph.prebuilt import create_react_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, ToolCall
from langchain_core.prompts import ChatPromptTemplate

from a2a.client import A2AClient
from a2a.types import Task, Message, Part, TextPart, AgentCard
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# SỬA LỖI: Định nghĩa schema cho công cụ send_message_to_agent
class SendMessageSchema(BaseModel):
    agent_name: str = Field(description="The name of the agent to send the message to.")
    query: str = Field(description="The query or message to send to the agent.")


class GPTOSSChatModel(BaseChatModel):
    base_url: str = Field(...)
    model: str = Field(...)
    system_prompt: str = ""
    _bound_tools: Optional[List[BaseTool]] = None

    def bind_tools(self, tools: List[BaseTool], **kwargs) -> "GPTOSSChatModel":
        clone = self.model_copy(deep=True)
        clone._bound_tools = tools
        return clone

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        api_messages = []
        if self.system_prompt:
            if not any(m.type == "system" for m in messages):
                api_messages.append({"role": "system", "content": self.system_prompt})

        for m in messages:
            if m.type == "system" and self.system_prompt:
                # Ghi đè system message mặc định bằng prompt của chúng ta
                # Đảm bảo chỉ có một system message
                if not any(d.get('role') == 'system' for d in api_messages):
                    api_messages.append({"role": "system", "content": self.system_prompt})
                continue

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
    SUPPORTED_CONTENT_TYPES = ['text/plain']

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

    def __init__(self, remote_agents: List[Dict[str, Any]]):
        if not remote_agents:
            raise RuntimeError("Could not discover any remote agents. Check URLs in .env")

        self.remote_agents = remote_agents
        self.tools = self._create_tools()

        system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
            agents_list=json.dumps(self.remote_agents, indent=2)
        )

        self.llm = GPTOSSChatModel(
            base_url=os.getenv("TOOL_LLM_URL"),
            model=os.getenv("TOOL_LLM_NAME"),
            system_prompt=system_prompt,
        )

        self.agent_executor = create_react_agent(self.llm, self.tools)

    @classmethod
    async def create(cls):
        """
        Phương thức bất đồng bộ để khám phá agent và khởi tạo một instance của HostAgent.
        """
        discovered_agents = await cls._discover_remote_agents()
        return cls(remote_agents=discovered_agents)

    @staticmethod
    async def _discover_remote_agents() -> List[Dict[str, Any]]:
        """
        Đọc AgentCard từ các URL trong .env và trả về danh sách thông tin agent.
        """
        agent_urls = [url.strip() for url in os.getenv("REMOTE_AGENT_URLS", "").split(",") if url.strip()]
        discovered_agents = []

        async with httpx.AsyncClient() as client:
            async def fetch_card(url: str):
                if not url.startswith(('http://', 'https://')):
                    url = 'http://' + url
                card_url = f"{url.rstrip('/')}{AGENT_CARD_WELL_KNOWN_PATH}"
                response = await client.get(card_url)
                response.raise_for_status()
                return AgentCard(**response.json())

            tasks = [fetch_card(url) for url in agent_urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for agent_card in results:
                if isinstance(agent_card, Exception):
                    logger.error(f"Failed to fetch agent card: {agent_card}")
                    continue

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
                    a2a_client = A2AClient(client, url=agent_info['url'])
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
            # SỬA LỖI: Cung cấp args_schema cho công cụ
            Tool(
                name="send_message_to_agent",
                func=send_message_to_agent_func,
                description=send_message_to_agent_func.__doc__,
                args_schema=SendMessageSchema,
            )
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
                    # Đoạn code này bây giờ sẽ chạy đúng vì LLM sẽ cung cấp key 'query'
                    content = f"Contacting agent `{tool_args['agent_name']}` with query: `{tool_args['query']}`\n"
                else:
                    content = f"Using tool `{tool_name}`...\n"
                yield {'is_task_complete': False, 'content': content}

            elif isinstance(last_message, ToolMessage):
                yield {'is_task_complete': False, 'content': f"Received result from tool `{last_message.name}`.\n"}

            elif isinstance(last_message, AIMessage) and not last_message.tool_calls:
                final_answer = last_message.content
                break

        yield {'is_task_complete': True, 'content': final_answer}