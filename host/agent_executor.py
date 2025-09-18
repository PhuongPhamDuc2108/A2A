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
    ContentType,
)
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError
from app.agent import HostAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HostAgentExecutor(AgentExecutor):
    """
    Executor to run the HostAgent.
    """

    def __init__(self):
        self.agent = HostAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Executes the agent's stream method and handles task updates.
        """
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
                # If the task is not yet complete, send a working status update
                if not item.get('is_task_complete'):
                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(item['content'], task.context_id, task.id),
                    )
                # If the task is complete, add the final result as an artifact
                else:
                    await updater.add_artifact(
                        [Part(root=TextPart(text=item['content']))],
                        name='final_result',
                    )
                    await updater.complete()
                    break
        except Exception as e:
            logger.error(f'An error occurred while executing the host agent: {e}', exc_info=True)
            # Fails the task and reports the internal error
            await updater.fail(error=InternalError(message=str(e)))
            raise ServerError(error=InternalError()) from e

    def _validate_request(self, context: RequestContext) -> bool:
        """
        Validates the incoming request. Returns True if invalid.
        """
        return False

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Handles cancellation of a task.
        """
        logger.info(f"Cancellation requested for task {context.current_task.id}")
        # Implement cancellation logic if needed, e.g., stopping the thread in the agent.
        pass

