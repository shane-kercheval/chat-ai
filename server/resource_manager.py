"""SQLite-backed resource manager with worker processes for content extraction."""
from enum import IntEnum
import json
import logging
import logging.config
import os
import sqlite3
from datetime import datetime, timezone
from multiprocessing import Process, Queue, Manager
from pathlib import Path
import asyncio
from typing import Any
import aiofiles
from cachetools import LRUCache
from dataclasses import dataclass
from proto.generated import chat_pb2
from server.agents.context_strategy_agent import (
    ContextStrategyAgent,
    ContextType,
)
from server.utilities import (
    clean_text_from_pdf,
    extract_jupyter_notebook_content,
    extract_text_from_pdf,
    generate_directory_tree,
    extract_html_from_webpage,
    clean_html_from_webpage,
    CODE_EXTENSIONS,
)
from server.vector_db import Document, SimilarityScorer
from dotenv import load_dotenv
load_dotenv()

logging.config.fileConfig('config/logging.conf')


class ContextStrategy(IntEnum):
    """Resource retrieval strategy that corresponds to the enum in `.proto`."""

    FULL_TEXT = 0
    RAG = 1
    AUTO = 2


class ResourceNotFoundError(Exception):
    """Raised when a resource path doesn't exist."""


@dataclass
class Resource:
    """Represents a stored resource."""

    path: str
    type: chat_pb2.ResourceType
    content: str | None
    last_accessed: str
    last_modified: str
    metadata: dict[str, Any] | None


def _get_last_modified_time(path: str | Path) -> str:
    """Get the last modified time of a file as an ISO string."""
    if isinstance(path, str):
        path = Path(path)
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _get_utc_now() -> str:
    """Get the current time as an ISO string."""
    return datetime.now(timezone.utc).isoformat()


async def _extract_file_content(path: str) -> tuple[str, str]:
    """Extract content from a file and get its last modified time."""
    file_path = Path(path)
    if not file_path.exists():
        raise ResourceNotFoundError(f"File not found: {path}")
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    # Get last modified time
    last_modified = _get_last_modified_time(file_path)
    if path.endswith('.pdf'):
        content = await extract_text_from_pdf(path)
        content = clean_text_from_pdf(content)
    elif path.endswith('.ipynb'):
        content = extract_jupyter_notebook_content(path)
    else:
        async with aiofiles.open(path) as f:
            content = await f.read()
    return content, last_modified


class ResourceWorker(Process):
    """Worker process for handling resource content extraction."""

    def __init__(self, queue: Queue, db_path: str):
        super().__init__()
        self.queue = queue
        self.db_path = db_path
        self.daemon = True

    def _process_resource(
            self,
            path: str,
            type: chat_pb2.ResourceType,  # noqa: A002
            metadata: dict | None,
        ) -> None:
        try:
            if type == chat_pb2.ResourceType.FILE:
                content, last_modified = asyncio.run(_extract_file_content(path))
            elif type == chat_pb2.ResourceType.DIRECTORY:
                # directory resources are processed on demand and not stored in the database, so
                # this function should not be called with directory resources
                raise ValueError("Directory resources should not be processed")
            elif type == chat_pb2.ResourceType.WEBPAGE:
                if 'arxiv.org/pdf/' in path:
                    content = asyncio.run(extract_text_from_pdf(path))
                    content = clean_text_from_pdf(content)
                else:
                    content = asyncio.run(extract_html_from_webpage(path))
                    content = clean_html_from_webpage(content)
                last_modified = _get_utc_now()
            else:
                raise ValueError(f"Invalid resource type: {type}")

            metadata_json = json.dumps(metadata) if metadata else None
            timestamp = _get_utc_now()

            with sqlite3.connect(self.db_path) as db:
                # Check if resource exists and compare content
                cursor = db.execute(
                    "SELECT content FROM resources WHERE path = ?",
                    (path,),
                )
                existing = cursor.fetchone()
                if existing:
                    if existing[0] != content:
                        # Content changed - update everything
                        db.execute(
                            """
                            UPDATE resources
                            SET content = ?, type = ?, last_accessed = ?, last_modified = ?, metadata = ?
                            WHERE path = ?
                            """,  # noqa: E501
                            (content, type, timestamp, last_modified, metadata_json, path),
                        )
                    else:
                        # Content unchanged - just update last_accessed
                        db.execute(
                            "UPDATE resources SET last_accessed = ? WHERE path = ?",
                            (timestamp, path),
                        )
                else:
                    # New resource - insert
                    db.execute(
                        """
                        INSERT INTO resources (path, type, content, last_accessed, last_modified, metadata)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,  # noqa: E501
                        (path, type, content, timestamp, last_modified, metadata_json),
                    )
                db.commit()
        except Exception as e:
            logging.error(f"Error processing resource {path}: {e}")

    def run(self) -> None:
        """Process resources from the queue."""
        while True:
            try:
                path, type, metadata, lock = self.queue.get()  # noqa: A001
                self._process_resource(path, type, metadata)
            except Exception as e:
                logging.error(f"Worker error: {e}")
            finally:
                lock.release()


class ResourceManager:
    """Manages resources with SQLite backing and worker processes."""

    def __init__(
            self,
            db_path: str,
            num_workers: int = 2,
            rag_scorer: SimilarityScorer | None = None,
            rag_char_threshold: int = 5000,
            context_strategy_model_config: dict | None = None,
        ):
        """
        Initialize the resource manager.

        Args:
            db_path:
                Path to SQLite database
            num_workers:
                Number of worker processes to start
            rag_scorer:
                SimilarityScorer object for semantic search
            rag_char_threshold:
                Minimum character count for RAG. (Documents below this threshold will automatically
                include the entire content in the context.)
            context_strategy_model_config:
                Model config to use for the strategy agent. Must contain `client_type` and
                `model_name` keys. Can contain additional keys that will be passed to the model
                (e.g. `temperature`).
        """
        self.db_path = db_path
        self._manager = Manager()
        self._global_lock = asyncio.Lock()
        self._rag_scorer = rag_scorer
        self._rag_char_threshold = rag_char_threshold
        self._context_strategy_model_config = context_strategy_model_config
        self._resource_locks = LRUCache(maxsize=1000)
        self._work_queue = Queue()
        self._workers = [
            ResourceWorker(queue=self._work_queue, db_path=db_path)
            for _ in range(num_workers)
        ]

    async def initialize(self) -> None:
        """Initialize the database and start workers."""
        # Initialize database
        with sqlite3.connect(self.db_path) as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS resources (
                    path TEXT PRIMARY KEY,
                    type INTEGER NOT NULL,
                    content TEXT,
                    last_accessed TIMESTAMP NOT NULL,
                    last_modified TIMESTAMP NOT NULL,
                    metadata TEXT
                )
            """)
            db.commit()
        # Start worker processes
        for worker in self._workers:
            worker.start()

    async def get_resource_lock(self, path: str) -> object:
        """Get a lock for a resource path."""
        async with self._global_lock:
            if path not in self._resource_locks:
                self._resource_locks[path] = self._manager.Lock()
            return self._resource_locks[path]

    async def add_resource(
        self,
        path: str,
        type: chat_pb2.ResourceType,  # noqa: A002
        metadata: dict | None = None,
    ) -> None:
        """
        Add or update a resource. This function will return immediately after adding the resource
        to the work queue. The resource will be processed by a worker process.

        `get_resource` should be used to retrieve the resource after adding it, which will block
        until the resource is processed.

        Args:
            path: Resource path
            type: Resource type (FILE, DIRECTORY, WEBPAGE)
            metadata: Optional metadata dict

        Note:
            If the resource is already being processed, the request is ignored.
        """
        if type == chat_pb2.ResourceType.DIRECTORY:
            # we do not store directory resources, the content is extracted on demand
            return
        if type == chat_pb2.ResourceType.FILE and not os.path.isfile(path):
            raise ResourceNotFoundError(f"File not found: {path}")

        resource_lock = await self.get_resource_lock(path)
        acquired = resource_lock.acquire(blocking=False)  # Try to acquire without blocking
        if acquired:
            self._work_queue.put((path, type, metadata, resource_lock))
        else:
            logging.error(f"Resource {path} is already being processed, ignoring add request")

    async def get_resource(self, path: str, type: chat_pb2.ResourceType) -> Resource:  # noqa: A002
        """
        Get a resource by path. This function will block if the resource is being processed.

        If the resource is a file, and the `last modified` timestamp has changed, the content will
        be updated (in the database) and the resource will be reprocessed before returning.

        Args:
            path:
                Resource path
            type:
                Resource type; provided so that the function can determine how to process the
                resource without needing to check the database.

        Returns:
            Resource object

        Raises:
            ResourceNotFoundError: If resource doesn't exist in database
            ValueError: If stored resource type is invalid
        """
        if type == chat_pb2.ResourceType.DIRECTORY:
            try:
                content = await generate_directory_tree(path)
            except ValueError as e:
                raise ResourceNotFoundError(f"Directory not found: {path}") from e
            return Resource(
                path=path,
                type=chat_pb2.ResourceType.DIRECTORY,
                content=content,
                last_accessed=_get_utc_now(),
                last_modified=_get_utc_now(),
                metadata=None,
            )

        resource_lock = await self.get_resource_lock(path)
        with resource_lock, sqlite3.connect(self.db_path) as db:
            cursor = db.execute(
                """
                    SELECT type, content, last_modified, metadata
                    FROM resources
                    WHERE path = ?
                    """,
                (path,),
            )
            result = cursor.fetchone()
            if not result:
                raise ResourceNotFoundError(f"Resource not found: {path}")

            resource_type, content, last_modified, metadata_json = result
            metadata = json.loads(metadata_json) if metadata_json else None

            def _update_last_accessed(timestamp, path):  # noqa
                db.execute(
                    "UPDATE resources SET last_accessed = ? WHERE path = ?",
                    (timestamp, path),
                )

            last_accessed = _get_utc_now()
            if resource_type == chat_pb2.ResourceType.FILE:
                if not os.path.isfile(path):
                    raise ResourceNotFoundError(f"File not found: {path}")
                # Check if file content has changed
                file_mtime = _get_last_modified_time(path)
                if last_modified != file_mtime:
                    logging.info(f"File content changed, updating: `{path}`")
                    content, last_modified = await _extract_file_content(path)
                    db.execute(
                        """
                        UPDATE resources
                        SET content = ?, last_accessed = ?, last_modified = ?
                        WHERE path = ?
                        """,
                        (content, last_accessed, last_modified, path),
                    )
                else:
                    _update_last_accessed(last_accessed, path)
            else:
                _update_last_accessed(last_accessed, path)
            db.commit()
            return Resource(
                path=path,
                type=resource_type,
                content=content,
                last_accessed=last_accessed,
                last_modified=last_modified,
                metadata=metadata,
            )

    async def create_context(  # noqa: PLR0912, PLR0915
            self,
            resources: list[chat_pb2.Resource],
            query: str | None = None,
            rag_similarity_threshold: float | None = None,
            rag_max_k: int | None = None,
            max_content_length: int | None = None,
            context_strategy: ContextStrategy | None = None,
        ) -> tuple[str, dict[str, ContextType]]:
        """
        Create the context that will be given to the model.

        Args:
            resources: List of resources to process
            query: Optional query string for semantic search
            rag_similarity_threshold: Minimum similarity score for a chunk to be included
            rag_max_k: Maximum number of chunks to include (not including neighbors)
            max_content_length: Maximum number of characters in the final context
            auto_extract:
                TBD.
                If True, uses an agent to decide if each file should be used, and if so, whether
                to use the full content or use RAG to extract relevant content.
            context_strategy:
                Strategy to use for resource retrieval. If `FULL_TEXT`, the full content of each
                file will be included in the context. If `RAG`, RAG will be used to extract
                relevant content, except for directories (which will use the entire tree
                structure), and code files (which will use the full content). If `AUTO`, the
                "Context Strategy Agent" will be used to determine the strategy for each file:
                which may include excluding the file, using the full content, or using RAG.

        Returns:
            Combined context string

        Raises:
            ResourceNotFoundError: If any resource cannot be found
        """
        if not resources:
            return "", {}

        if rag_similarity_threshold is None:
            rag_similarity_threshold = 0.5
        if context_strategy is None:
            context_strategy = ContextStrategy.RAG

        # dedupe the list of resources by path
        resources = list({r.path: r for r in resources}.values())
        # convert list of chat_pb2.Resource to list of Resource objects
        resources = await asyncio.gather(*(self.get_resource(r.path, r.type) for r in resources))

        if context_strategy == ContextStrategy.AUTO and not self._context_strategy_model_config:
            raise ValueError("Context strategy model must be provided for AUTO strategy")
        if context_strategy == ContextStrategy.AUTO and not query:
            raise ValueError("Query must be provided for AUTO strategy")

        def use_full_text(resource: Resource) -> bool:
            if resource.type == chat_pb2.ResourceType.DIRECTORY:
                return True
            if len(resource.content) <= self._rag_char_threshold:
                return True
            if resource.type == chat_pb2.ResourceType.FILE:
                return Path(resource.path).suffix in CODE_EXTENSIONS
            return False

        # build context strategies which is a list of tuples (resource object and the strategy)
        if context_strategy == ContextStrategy.FULL_TEXT:
            context_types_used = [(r, ContextType.FULL_TEXT) for r in resources]
        elif context_strategy == ContextStrategy.RAG:
            context_types_used = [
                (r, ContextType.FULL_TEXT if use_full_text(r) else ContextType.RAG)
                for r in resources
            ]
        elif context_strategy == ContextStrategy.AUTO:
            # Get context strategy from agent for each file
            agent = ContextStrategyAgent(**self._context_strategy_model_config)
            summary = await agent(
                messages=[{"role": "user", "content": query}],
                resource_names=[r.path for r in resources],
            )
            context_types_used = []
            for s, r in zip(summary.strategies, resources, strict=True):
                if s.resource_name not in r.path:  # agent might use only file name without path
                    logging.error(f"Mismatched file names: `{s.resource_name}` vs `{r.path}`")
                    logging.error(f"Mismatch - Provided Resources: {[r.path for r in resources]}")
                    logging.error(f"Mismatch - Returned Strategies: {summary.strategies}")
                # if the strategy is to use RAG then check if we need to override the
                # agent's recommendation based on the file type/content.
                if s.context_type == ContextType.RAG and use_full_text(r):
                    context_types_used.append((r, ContextType.FULL_TEXT))
                else:
                    context_types_used.append((r, s.context_type))
        else:
            raise ValueError(f"Invalid resource strategy: {context_strategy}")

        if not context_types_used or len(context_types_used) != len(resources):
            raise ValueError("Invalid context strategies")

        contexts = []
        # Collect documents that need similarity search
        rag_resources = []
        # for each resource, either ignore, get full context, or add to the list of docs we need to
        # do RAG on
        for resource, strategy in context_types_used:
            if strategy == ContextType.IGNORE:
                continue
            if strategy == ContextType.FULL_TEXT:
                contexts.append(f"Content from `{resource.path}`:\n\n```\n{resource.content}\n```")
            elif strategy == ContextType.RAG:
                rag_resources.append(Document(
                    key=resource.path,
                    content=resource.content,
                    last_modified=resource.last_modified,
                ))
            else:
                raise ValueError(f"Invalid context strategy: {strategy}")

        # Process documents that need similarity search
        if rag_resources:
            if not (self._rag_scorer and query):
                raise ValueError("RAG scorer and query must be provided for RAG strategy")

            # returns list of objects containing chunk/score
            search_results = self._rag_scorer.score(rag_resources, query)
            # Group results by document
            results_by_doc = {}
            for result in search_results:
                if result.key not in results_by_doc:
                    results_by_doc[result.key] = []
                results_by_doc[result.key].append(result)

            # Process each document's results
            for doc_key, doc_results in results_by_doc.items():
                # Filter chunks above threshold
                above_threshold = [r for r in doc_results if r.score >= rag_similarity_threshold]
                # Apply rag_max_k if specified
                if rag_max_k is not None and above_threshold:
                    # Sort by score and take top k
                    above_threshold.sort(key=lambda x: x.score, reverse=True)
                    above_threshold = above_threshold[:rag_max_k]
                    # Resort by index for processing
                    above_threshold.sort(key=lambda x: x.index)

                relevant_chunks = []
                included_indices = set()

                # Process chunks and their neighbors
                for result in above_threshold:
                    idx = doc_results.index(result)
                    # Add previous chunk if exists and not already included
                    if idx > 0 and doc_results[idx-1].index not in included_indices:
                        relevant_chunks.append(doc_results[idx-1])
                        included_indices.add(doc_results[idx-1].index)
                    # Add current chunk if not already included
                    if result.index not in included_indices:
                        relevant_chunks.append(result)
                        included_indices.add(result.index)
                    # Add next chunk if exists and not already included
                    if idx < len(doc_results)-1:
                        next_result = doc_results[idx+1]
                        if next_result.index not in included_indices:
                            relevant_chunks.append(next_result)
                            included_indices.add(next_result.index)

                # Sort final chunks by index
                relevant_chunks.sort(key=lambda x: x.index)

                # If we found relevant chunks, build the context
                if relevant_chunks:
                    doc_context = [f"Content from `{doc_key}`:\n\n```"]
                    last_index = -1
                    for result in relevant_chunks:
                        # Add ... if there's a gap
                        if last_index != -1 and result.index > last_index + 1:
                            doc_context.append("...")
                        doc_context.append(result.text)
                        last_index = result.index
                    # add ... to end if there's a gap
                    if last_index != len(doc_results) - 1:
                        doc_context.append("...")
                    doc_context.append("```")
                    doc_context = '\n'.join(doc_context)
                    contexts.append(doc_context)

        final_context = "\n\n".join(contexts)
        # Apply max_content_length if specified
        if max_content_length is not None and len(final_context) > max_content_length:
            final_context = final_context[-max_content_length:]

        return final_context, {r.path: s for r, s in context_types_used}

    async def shutdown(self) -> None:
        """Shutdown worker processes."""
        self._manager.shutdown()
        for worker in self._workers:
            worker.terminate()
            worker.join()
