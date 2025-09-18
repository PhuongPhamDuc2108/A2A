import logging
import os
import sys
import httpx
import click
import uvicorn
from dotenv import load_dotenv

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
# SỬA LỖI: Import thêm các thành phần cần thiết cho push notification
from a2a.server.tasks import (
    InMemoryTaskStore,
    InMemoryPushNotificationConfigStore,
    BasePushNotificationSender,
)
from a2a.types import AgentCard, AgentCapabilities, AgentSkill

from calculator_agent import CalculatorAgent, GPTOSSChatModel, calculator_tool
from calculator_agent_executor import CalculatorAgentExecutor

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingEnvVarError(Exception):
    """Lỗi thiếu biến môi trường."""


@click.command()
@click.option('--host', default='localhost', help='Host to run the server on.')
@click.option('--port', default=10001, help='Port to run the server on.')
def main(host, port):
    """Khởi động máy chủ cho Calculator Agent."""
    try:
        if not os.getenv('TOOL_LLM_URL'):
            raise MissingEnvVarError('TOOL_LLM_URL environment variable not set.')
        if not os.getenv('TOOL_LLM_NAME'):
            raise MissingEnvVarError('TOOL_LLM_NAME environment variable not set.')

        # SỬA LỖI: Thêm push_notifications=False để rõ ràng hơn
        capabilities = AgentCapabilities(streaming=True, push_notifications=False)
        skill = AgentSkill(
            id='calculator',
            name='Numerical Calculator Tool',
            description='Performs numerical calculations based on a given expression.',
            tags=['calculator', 'math', 'arithmetic'],
            examples=['What is 100 * (5 + 2)?', 'Calculate 9^3'],
        )

        agent_card = AgentCard(
            name='Calculator Agent',
            description='An agent that can perform mathematical calculations.',
            url=f'http://{host}:{port}/',
            version='1.0.0',
            default_input_modes=['text/plain'],
            default_output_modes=['text/plain'],
            capabilities=capabilities,
            skills=[skill],
        )

        llm = GPTOSSChatModel(
            base_url=os.getenv("TOOL_LLM_URL"),
            model=os.getenv("TOOL_LLM_NAME")
        )
        tools = [calculator_tool]
        agent = CalculatorAgent(llm=llm, tools=tools)

        # --- SỬA LỖI: BẮT ĐẦU ---
        # Thêm cấu hình cho Push Notification (ngay cả khi không dùng)
        push_config_store = InMemoryPushNotificationConfigStore()
        client = httpx.AsyncClient()

        # 2. Truyền client vào push_sender
        push_sender = BasePushNotificationSender(
            config_store=push_config_store, httpx_client=client
        )

        request_handler = DefaultRequestHandler(
            agent_executor=CalculatorAgentExecutor(agent=agent),
            task_store=InMemoryTaskStore(),
            # Truyền các tham số còn thiếu vào
            push_config_store=push_config_store,
            push_sender=push_sender,
        )
        # --- SỬA LỖI: KẾT THÚC ---

        server = A2AStarletteApplication(
            agent_card=agent_card, http_handler=request_handler
        )

        uvicorn.run(server.build(), host=host, port=port)

    except MissingEnvVarError as e:
        logger.error(f'Configuration Error: {e}')
        sys.exit(1)
    except Exception as e:
        logger.error(f'An error occurred during server startup: {e}', exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

