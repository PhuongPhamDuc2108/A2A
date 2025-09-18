import logging
import os
import sys

import click
import uvicorn
from dotenv import load_dotenv

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

# Đảm bảo import đúng agent và executor cho Search
from search_agent import SearchAgent
from search_agent_executor import SearchAgentExecutor

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingEnvVarError(Exception):
    """Exception for missing environment variables."""


@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=10002)  # Cổng mặc định cho Search Agent
def main(host, port):
    """Starts the Search Agent server."""
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

        request_handler = DefaultRequestHandler(
            agent_executor=SearchAgentExecutor(),
            task_store=InMemoryTaskStore(),
        )

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

