"""Provides a class for merging async generators into a single stream."""
import asyncio
from collections.abc import Generator


class AsyncMerge:
    """
    Merges multiple async generators into a single stream, yielding items as they arrive.

    AsyncMerge runs all generators concurrently and maintains their individual speeds -
    fast generators aren't slowed down by slow ones. Items are yielded as soon as they
    become available from any generator.

    Example:
        async def fast_gen():
            for i in range(100):
                yield f"fast-{i}"

        async def slow_gen():
            for i in range(5):
                await asyncio.sleep(0.1)
                yield f"slow-{i}"

        merger = AsyncMerge([fast_gen(), slow_gen()])
        async for item in merger:
            print(item)  # Items interleaved as they arrive
    """

    def __init__(self, generators: list[Generator]):
        """
        Initialize AsyncMerge with a list of generators to merge.

        Args:
            generators: List of async generators to merge into a single stream
        """
        self.generators = generators
        self.queue = asyncio.Queue()

    async def __aiter__(self):
        """
        Iterate over merged stream of items from all generators.

        Yields items as they become available from any generator. When all generators are
        exhausted, the iteration ends.

        Yields:
            Items from any generator in the order they were received
        """
        # Start all generators running independently
        tasks = [
            asyncio.create_task(self._forward(gen))
            for gen in self.generators
        ]
        try:
            # Keep going while we have either tasks running or items in queue
            while tasks or not self.queue.empty():
                wait_tasks = list(tasks)  # Copy task list
                # Only wait on queue if we have tasks still running
                if tasks:
                    get_task = asyncio.create_task(self.queue.get())
                    wait_tasks.append(get_task)
                    done, _ = await asyncio.wait(
                        wait_tasks,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                else:
                    # Just drain remaining queue items
                    yield await self.queue.get()
                    continue
                # Process completed tasks
                for task in done:
                    if task is get_task:
                        # Item received from queue
                        yield await task
                    else:
                        # Generator task completed
                        if task.exception() is not None:
                            raise task.exception()
                        tasks.remove(task)
                if get_task not in done:
                    get_task.cancel()
        finally:
            for task in tasks:
                task.cancel()

    async def _forward(self, generator: Generator) -> None:
        """
        Forward items from a generator to the shared queue.

        Runs continuously until the generator is exhausted, putting each item
        into the queue as it arrives. Sends None to signal completion.

        Args:
            generator: Async generator to forward items from
        """
        async for item in generator:
            await self.queue.put(item)
