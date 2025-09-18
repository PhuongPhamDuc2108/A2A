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
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from search_agent import SearchAgent, GPTOSSChatModel, search_tool
from search_agent_executor import SearchAgentExecutor

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ĐỊNH NGHĨA AGENT CARD Ở PHẠM VI TOÀN CỤC
agent_card: AgentCard = None


class MissingEnvVarError(Exception):
    """Exception for missing environment variables."""


# HÀM MỚI ĐỂ PHỤC VỤ AGENT CARD
async def serve_agent_card(request):
    """Endpoint để trả về thông tin Agent Card."""
    if agent_card:
        return JSONResponse(agent_card.model_dump(mode='json'))
    return JSONResponse({'error': 'Agent card not initialized'}, status_code=500)


@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=10002)
def main(host, port):
    """Starts the Search Agent server."""
    global agent_card
    try:
        # Kiểm tra các biến môi trường cần thiết
        if not os.getenv('TOOL_LLM_URL'):
            raise MissingEnvVarError('TOOL_LLM_URL environment variable not set.')
        if not os.getenv('TOOL_LLM_NAME'):
            raise MissingEnvVarError('TOOL_LLM_NAME environment variable not set.')
        if not os.getenv('TAVILY_API_KEY'):
            raise MissingEnvVarError('TAVILY_API_KEY environment variable not set.')

        capabilities = AgentCapabilities(streaming=True, push_notifications=False)
        skill = AgentSkill(
            id='web_search',
            name='Internet Search Tool (Tavily)',
            description='Helps find up-to-date information on the internet.',
            tags=['search', 'web', 'tavily', 'information'],
            examples=['Who won the world cup in 2022?', 'What is the weather in Hanoi?'],
        )

        # Gán giá trị cho biến agent_card toàn cục
        agent_card = AgentCard(
            name='Search Agent',
            description='An agent that can search the internet for information.',
            url=f'http://{host}:{port}/',
            version='1.0.0',
            default_input_modes=['text/plain'],
            default_output_modes=['text/plain'],
            capabilities=capabilities,
            skills=[skill],
        )

        # Khởi tạo LLM và Agent
        llm = GPTOSSChatModel(
            base_url=os.getenv("TOOL_LLM_URL"),
            model=os.getenv("TOOL_LLM_NAME")
        )
        tools = [search_tool]
        agent = SearchAgent(llm=llm, tools=tools)

        # Đảm bảo SearchAgentExecutor nhận agent khi khởi tạo
        # (Bạn cần sửa constructor của SearchAgentExecutor để nhận 'agent')
        request_handler = DefaultRequestHandler(
            agent_executor=SearchAgentExecutor(agent=agent),
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