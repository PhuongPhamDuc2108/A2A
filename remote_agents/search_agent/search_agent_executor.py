import logging
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InternalError,
    InvalidParamsError,
    Part,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError
from search_agent import SearchAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SearchAgentExecutor(AgentExecutor):
    """Search Agent Executor."""

    def __init__(self,agent:SearchAgent):
        self.agent = agent

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if self._validate_request(context):
            raise ServerError(error=InvalidParamsError())

        query = context.get_user_input()
        if not query:
            raise ServerError(error=InvalidParamsError(message="Input query cannot be empty."))

        task = context.current_task or new_task(context.message)
        await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)

        try:
            async for item in self.agent.stream(query, task.context_id):
                if not item['is_task_complete']:
                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(item['content'], task.context_id, task.id),
                    )
                else:
                    await updater.add_artifact(
                        [Part(root=TextPart(text=item['content']))],
                        name='search_result',
                    )
                    await updater.complete()
                    break
        except Exception as e:
            logger.error(f'An error occurred while executing the agent: {e}', exc_info=True)
            await updater.fail(error=InternalError(message=str(e)))
            raise ServerError(error=InternalError()) from e

    def _validate_request(self, context: RequestContext) -> bool:
        return False

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())

