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

from app.agent_executor import HostAgentExecutor
from app.agent import HostAgent

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingEnvVarError(Exception):
    """Custom exception for missing environment variables."""


@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=10000, help="Port for the Host Agent server.")
def main(host, port):
    """
    Starts the Host Agent server which routes tasks to remote agents.
    """
    try:
        # Check for required environment variables
        if not os.getenv('TOOL_LLM_URL'):
            raise MissingEnvVarError('TOOL_LLM_URL environment variable not set.')
        if not os.getenv('TOOL_LLM_NAME'):
            raise MissingEnvVarError('TOOL_LLM_NAME environment variable not set.')
        if not os.getenv('REMOTE_AGENT_URLS'):
            raise MissingEnvVarError(
                'REMOTE_AGENT_URLS environment variable not set (e.g., "http://host:port1,http://host:port2").')

        # Define agent capabilities and skills for the Agent Card
        capabilities = AgentCapabilities(streaming=True, push_notifications=False)
        skill = AgentSkill(
            id='task_dispatcher',
            name='Task Dispatcher',
            description='Can delegate tasks like math calculations or web searches to specialized agents.',
            tags=['dispatcher', 'router', 'orchestrator', 'assistant'],
            examples=['What is (100 / 5) ^ 2?', 'Who won the latest world cup?'],
        )
        agent_card = AgentCard(
            name='Host Agent',
            description='A central agent that intelligently routes queries to other specialized agents.',
            url=f'http://{host}:{port}/',
            version='1.0.0',
            default_input_modes=HostAgent.SUPPORTED_CONTENT_TYPES,
            default_output_modes=HostAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        # Set up the request handler with the agent executor
        request_handler = DefaultRequestHandler(
            agent_executor=HostAgentExecutor(),
            task_store=InMemoryTaskStore(),
        )

        # Create the A2A server application
        server = A2AStarletteApplication(
            agent_card=agent_card, http_handler=request_handler
        )

        # Run the server
        uvicorn.run(server.build(), host=host, port=port)

    except MissingEnvVarError as e:
        logger.error(f'Configuration Error: {e}')
        sys.exit(1)
    except Exception as e:
        logger.error(f'An unexpected error occurred during server startup: {e}', exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

