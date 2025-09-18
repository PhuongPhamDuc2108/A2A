import logging
import os
import sys
import click
import uvicorn
from dotenv import load_dotenv

# THÊM CÁC IMPORT CẦN THIẾT
from starlette.responses import JSONResponse
from starlette.routing import Route

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentCapabilities, AgentSkill

from calculator_agent import CalculatorAgent, GPTOSSChatModel, calculator_tool
from calculator_agent_executor import CalculatorAgentExecutor

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ĐỊNH NGHĨA AGENT CARD Ở PHẠM VI TOÀN CỤC ĐỂ CÓ THỂ TRUY CẬP
agent_card: AgentCard = None

class MissingEnvVarError(Exception):
    """Lỗi thiếu biến môi trường."""

# HÀM MỚI ĐỂ PHỤC VỤ AGENT CARD
async def serve_agent_card(request):
    """Endpoint để trả về thông tin Agent Card."""
    if agent_card:
        return JSONResponse(agent_card.model_dump(mode='json'))
    return JSONResponse({'error': 'Agent card not initialized'}, status_code=500)


@click.command()
@click.option('--host', default='localhost', help='Host to run the server on.')
@click.option('--port', default=10001, help='Port to run the server on.')
def main(host, port):
    """Khởi động máy chủ cho Calculator Agent."""
    global agent_card
    try:
        if not os.getenv('TOOL_LLM_URL'):
            raise MissingEnvVarError('TOOL_LLM_URL environment variable not set.')
        if not os.getenv('TOOL_LLM_NAME'):
            raise MissingEnvVarError('TOOL_LLM_NAME environment variable not set.')

        capabilities = AgentCapabilities(streaming=True, push_notifications=False)
        skill = AgentSkill(
            id='calculator',
            name='Numerical Calculator Tool',
            description='Performs numerical calculations based on a given expression.',
            tags=['calculator', 'math', 'arithmetic'],
            examples=['What is 100 * (5 + 2)?', 'Calculate 9^3'],
        )

        # Gán giá trị cho biến agent_card toàn cục
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

        request_handler = DefaultRequestHandler(
            agent_executor=CalculatorAgentExecutor(agent=agent),
            task_store=InMemoryTaskStore(),
        )

        server = A2AStarletteApplication(
            agent_card=agent_card, http_handler=request_handler
        )

        # Lấy ứng dụng starlette và thêm route thủ công
        app = server.build()
        app.routes.insert(
            0, Route("/.well-known/a2a/agent-card", endpoint=serve_agent_card, methods=["GET"])
        )

        uvicorn.run(app, host=host, port=port)

    except MissingEnvVarError as e:
        logger.error(f'Configuration Error: {e}')
        sys.exit(1)
    except Exception as e:
        logger.error(f'An error occurred during server startup: {e}', exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()