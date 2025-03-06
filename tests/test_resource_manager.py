"""Tests for ResourceManager."""
import asyncio
from textwrap import dedent
import time
import os
from pathlib import Path
import pytest
import aiosqlite
import tempfile
import aiofiles
import numpy as np

from proto.generated import chat_pb2
from server.agents.context_strategy_agent import (
    ContextStrategy as MockContextStrategy,
    ContextStrategies as MockContextStrategies,
    ContextType,
)
from server.resource_manager import ResourceManager, ResourceNotFoundError, ContextStrategy
from server.vector_db import SimilarityScorer
from tests.conftest import SKIP_CI, create_temp_file

def create_temp_db_path() -> str:
    """Create a temporary database file."""
    f = tempfile.NamedTemporaryFile(suffix='.db', delete=False)  # noqa: SIM115
    f.close()
    return f.name


@pytest.mark.asyncio
class TestResourceManager:
    """Tests for the ResourceManager."""

    async def test__initialization(self):
        """Test database initialization."""
        try:
            db_path = create_temp_db_path()
            manager = ResourceManager(db_path)
            await manager.initialize()
            assert os.path.exists(db_path)
            async with aiosqlite.connect(db_path) as db:
                cursor = await db.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name='resources'
                """)
                result = await cursor.fetchone()
                assert result is not None
                assert result[0] == "resources"
                # Verify table structure
                cursor = await db.execute("PRAGMA table_info(resources)")
                columns = await cursor.fetchall()
                column_names = [col[1] for col in columns]
                assert "path" in column_names
                assert "type" in column_names
                assert "content" in column_names
                assert "last_accessed" in column_names
                assert "last_modified" in column_names
                assert "metadata" in column_names
        finally:
            Path(db_path).unlink(missing_ok=True)

    async def test__add_resource__file__success(self):
        """Test successfully adding a file resource."""
        try:
            db_path = create_temp_db_path()
            file_path = create_temp_file("Test content")

            manager = ResourceManager(db_path)
            await manager.initialize()
            await manager.add_resource(
                path=file_path,
                type=chat_pb2.ResourceType.FILE,
                metadata={'encoding': 'utf-8'},
            )
            resource = await manager.get_resource(file_path, type=chat_pb2.ResourceType.FILE)
            assert resource
            assert resource.path == file_path
            assert resource.content == "Test content"
            assert resource.type == chat_pb2.ResourceType.FILE
            assert resource.last_accessed
            assert resource.last_modified
            assert 'encoding' in resource.metadata
            assert resource.metadata['encoding'] == 'utf-8'
            # Verify in database
            async with aiosqlite.connect(db_path) as db:
                cursor = await db.execute(
                    "SELECT type, content, last_modified, last_accessed, metadata FROM resources WHERE path = ?",  # noqa: E501
                    (file_path,),
                )
                result = await cursor.fetchone()
                assert result is not None
                assert result[0] == chat_pb2.ResourceType.FILE
                assert result[1] == "Test content"
                assert isinstance(result[2], str)
                assert isinstance(result[3], str)
                assert 'encoding' in result[4]
                assert 'utf-8' in result[4]
        finally:
            await manager.shutdown()
            Path(db_path).unlink(missing_ok=True)
            Path(file_path).unlink(missing_ok=True)

    async def test__add_resource__file__update_content(self):
        """Test updating a file resource when content changes."""
        try:
            db_path = create_temp_db_path()
            file_path = create_temp_file("Test content")

            manager = ResourceManager(db_path)
            await manager.initialize()
            await manager.add_resource(
                path=file_path,
                type=chat_pb2.ResourceType.FILE,
            )
            # blocks until worker processes the resource
            _ = await manager.get_resource(file_path, type=chat_pb2.ResourceType.FILE)
            # Get initial timestamps
            async with aiosqlite.connect(db_path) as db:
                cursor = await db.execute(
                    "SELECT last_accessed, last_modified FROM resources WHERE path = ?",
                    (file_path,),
                )
                initial_accessed, initial_modified = await cursor.fetchone()
            assert initial_accessed is not None
            assert initial_modified is not None
            # Update file content
            async with aiofiles.open(file_path, mode='w') as f:
                await f.write("Updated content")
            # Update resource
            await manager.add_resource(
                path=file_path,
                type=chat_pb2.ResourceType.FILE,
            )
            # blocks until worker processes the resource
            _ = await manager.get_resource(file_path, type=chat_pb2.ResourceType.FILE)
            # ensure there is only one row in the database
            async with aiosqlite.connect(db_path) as db:
                cursor = await db.execute(
                    "SELECT COUNT(*) FROM resources WHERE path = ?",
                    (file_path,),
                )
                count = await cursor.fetchone()
                assert count[0] == 1
            # # Verify updates
            async with aiosqlite.connect(db_path) as db:
                cursor = await db.execute(
                    "SELECT content, last_accessed, last_modified FROM resources WHERE path = ?",
                    (file_path,),
                )
                content, new_accessed, new_modified = await cursor.fetchone()
            assert content == "Updated content"
            assert new_accessed > initial_accessed
            assert new_modified > initial_modified
        finally:
            await manager.shutdown()
            Path(db_path).unlink(missing_ok=True)
            Path(file_path).unlink(missing_ok=True)

    async def test__add_resource__file__update_access_time_only(self):
        """Test that only last_accessed is updated when content hasn't changed."""
        try:
            db_path = create_temp_db_path()
            file_path = create_temp_file("Test content")

            manager = ResourceManager(db_path)
            await manager.initialize()
            await manager.add_resource(
                path=file_path,
                type=chat_pb2.ResourceType.FILE,
            )
            # blocks until worker processes the resource
            _ = await manager.get_resource(file_path, type=chat_pb2.ResourceType.FILE)
            async with aiosqlite.connect(db_path) as db:
                cursor = await db.execute(
                    "SELECT content, last_accessed, last_modified FROM resources WHERE path = ?",
                    (file_path,),
                )
                initial_content, initial_accessed, initial_modified = await cursor.fetchone()

            # Update resource without changing content
            await manager.add_resource(
                path=file_path,
                type=chat_pb2.ResourceType.FILE,
            )
            await asyncio.sleep(0.2)
            # Verify only last_accessed changed
            async with aiosqlite.connect(db_path) as db:
                cursor = await db.execute(
                    "SELECT content, last_accessed, last_modified FROM resources WHERE path = ?",
                    (file_path,),
                )
                new_content, new_accessed, new_modified = await cursor.fetchone()
                assert new_content == initial_content
                assert new_accessed > initial_accessed
                assert new_modified == initial_modified
        finally:
            await manager.shutdown()
            Path(db_path).unlink(missing_ok=True)
            Path(file_path).unlink(missing_ok=True)

    async def test__add_resource__file__with_metadata(self):
        """Test adding a file resource with metadata."""
        try:
            db_path = create_temp_db_path()
            file_path = create_temp_file("Test content")

            manager = ResourceManager(db_path)
            await manager.initialize()
            metadata = {'encoding': 'utf-8', 'language': 'python'}
            await manager.add_resource(
                path=file_path,
                type=chat_pb2.ResourceType.FILE,
                metadata=metadata,
            )
            # blocks until worker processes the resource
            _ = await manager.get_resource(file_path, type=chat_pb2.ResourceType.FILE)
            # Verify metadata was stored
            async with aiosqlite.connect(db_path) as db:
                cursor = await db.execute(
                    "SELECT metadata FROM resources WHERE path = ?",
                    (file_path,),
                )
                result = await cursor.fetchone()
            assert result is not None
            stored_metadata = result[0]
            assert 'encoding' in stored_metadata
            assert 'utf-8' in stored_metadata
            assert 'language' in stored_metadata
            assert 'python' in stored_metadata
        finally:
            await manager.shutdown()
            Path(db_path).unlink(missing_ok=True)
            Path(file_path).unlink(missing_ok=True)

    async def test__add_resource__file__not_found(self):
        """Test adding a nonexistent file."""
        try:
            db_path = create_temp_db_path()
            manager = ResourceManager(db_path)
            await manager.initialize()
            with pytest.raises(ResourceNotFoundError):
                await manager.add_resource(
                    path="/nonexistent/file.txt",
                    type=chat_pb2.ResourceType.FILE,
                )
            await asyncio.sleep(0.1)
            # Verify resource wasn't added
            async with aiosqlite.connect(db_path) as db:
                cursor = await db.execute(
                    "SELECT COUNT(*) FROM resources WHERE path = ?",
                    ("/nonexistent/file.txt",),
                )
                count = await cursor.fetchone()
                assert count[0] == 0
        finally:
            await manager.shutdown()
            Path(db_path).unlink(missing_ok=True)

    async def test__add_resource__file__not_a_file(self):
        """Test adding a path that exists but isn't a file."""
        try:
            db_path = create_temp_db_path()
            temp_dir = tempfile.mkdtemp()
            manager = ResourceManager(db_path)
            await manager.initialize()
            with pytest.raises(ResourceNotFoundError):
                await manager.add_resource(
                    path=temp_dir,
                    type=chat_pb2.ResourceType.FILE,
                )
        finally:
            Path(db_path).unlink(missing_ok=True)
            Path(temp_dir).rmdir()

    async def test__add_resource__invalid_type(self):
        """Test adding a resource with an invalid type."""
        try:
            db_path = create_temp_db_path()
            manager = ResourceManager(db_path)
            await manager.initialize()
            await manager.add_resource(
                path="/some/path",
                type=999,  # Invalid type
            )
        finally:
            await manager.shutdown()
            Path(db_path).unlink(missing_ok=True)

    async def test__concurrent_adds_ignore_duplicates(self):
        """Test that concurrent adds of the same resource are ignored."""
        try:
            db_path = create_temp_db_path()
            file_path = create_temp_file("test content")
            manager = ResourceManager(db_path, num_workers=2)
            await manager.initialize()

            # Create multiple add tasks for the same resource
            tasks = []
            for _ in range(10):
                task = asyncio.create_task(
                    manager.add_resource(file_path, chat_pb2.ResourceType.FILE),
                )
                tasks.append(task)

            # Wait for all tasks
            await asyncio.gather(*tasks)
            # blocks until worker processes the resource
            _ = await manager.get_resource(file_path, type=chat_pb2.ResourceType.FILE)

            async with aiosqlite.connect(db_path) as db:
                cursor = await db.execute("SELECT COUNT(*) FROM resources WHERE path = ?", (file_path,))  # noqa: E501
                count = await cursor.fetchone()
                assert count[0] == 1

        finally:
            await manager.shutdown()
            Path(db_path).unlink(missing_ok=True)
            Path(file_path).unlink(missing_ok=True)

    async def test__get_resource__large_file_blocks_successful(self):
        """Test that reads are blocked while resource is being processed."""
        try:
            db_path = create_temp_db_path()
            large_content = 'x' * 10_000_000  # 10MB of data
            file_path = create_temp_file(large_content)
            manager = ResourceManager(db_path, num_workers=1)
            await manager.initialize()
            await manager.add_resource(file_path, chat_pb2.ResourceType.FILE)
            result = await manager.get_resource(file_path, type=chat_pb2.ResourceType.FILE)
            assert result
            assert result.path == file_path
            assert result.content == large_content
            assert result.type == chat_pb2.ResourceType.FILE
            assert result.last_accessed
            assert result.last_modified
            assert not result.metadata
        finally:
            await manager.shutdown()
            Path(db_path).unlink(missing_ok=True)
            Path(file_path).unlink(missing_ok=True)

    async def test__add_resource__returns_immediately__get_resource__blocks(self):
        """Test that add_resource returns immediately while get_resource blocks."""
        try:
            num_files = 500
            db_path = create_temp_db_path()
            # Create multiple files
            files = [create_temp_file(f"content {i}") for i in range(num_files)]
            manager = ResourceManager(db_path, num_workers=2)
            await manager.initialize()

            # lets put all resource locks in cache so that we reduce delay in add_resource from
            # acquiring lock
            tasks = [
                asyncio.create_task(manager.get_resource_lock(f))
                for f in files
            ]
            await asyncio.gather(*tasks)
            assert len(manager._resource_locks) == num_files
            # Add all files concurrently
            start_time = time.time()
            tasks = [
                asyncio.create_task(
                    manager.add_resource(str(f), chat_pb2.ResourceType.FILE),
                )
                for f in files
            ]
            await asyncio.gather(*tasks)
            duration = time.time() - start_time
            # add_resource returns immediately after adding to queue so delay should be minimal
            # even with 500 files (we pre-created locks and added them to cache)
            assert duration < 0.15
            # and we should have less than num_files resources in the database
            async with aiosqlite.connect(db_path) as db:
                cursor = await db.execute("SELECT COUNT(*) FROM resources")
                count = await cursor.fetchone()
                assert count[0] < num_files
            # now let's test that get_resource blocks
            tasks = [
                asyncio.create_task(manager.get_resource(f, type=chat_pb2.ResourceType.FILE))
                for f in files
            ]
            results = await asyncio.gather(*tasks)
            for i, r in enumerate(results):
                assert r.content == f"content {i}"
                assert r.path == files[i]
                Path(files[i]).unlink(missing_ok=True)

            # now we should have num_files resources in the database
            async with aiosqlite.connect(db_path) as db:
                cursor = await db.execute("SELECT COUNT(*) FROM resources")
                count = await cursor.fetchone()
                assert count[0] == num_files
        finally:
            await manager.shutdown()
            Path(db_path).unlink(missing_ok=True)

    # async def test__lock_cleanup_after_processing(self):
    #     """Test that locks are properly cleaned up after processing."""
    #     try:
    #         db_path = create_temp_db_path()
    #         file_path = create_temp_file("test content")
    #         manager = ResourceManager(db_path, num_workers=1)
    #         await manager.initialize()
    #         await asyncio.sleep(1)

    #         async def lock_was_already_acquired(path: str, release: bool) -> bool:
    #             lock = await manager.get_resource_lock(path)
    #             has_acquired = lock.acquire(blocking=False)  # Try to acquire without blocking
    #             if release:
    #                 lock.release()
    #             # if we have acquired the lock, it means it was not already acquired
    #             return not has_acquired

    #         assert await lock_was_already_acquired(file_path, release=True) is False
    #         # Process resource
    #         await manager.add_resource(file_path, chat_pb2.ResourceType.FILE)
    #         assert await lock_was_already_acquired(file_path, release=False) is True
    #         await asyncio.sleep(1.0)
    #         assert await lock_was_already_acquired(file_path, release=True) is False
    #         # Process again - should work if lock is properly released
    #         await manager.add_resource(file_path, chat_pb2.ResourceType.FILE)
    #         resource = await manager.get_resource(file_path, type=chat_pb2.ResourceType.FILE)
    #         assert resource.content == "test content"
    #     finally:
    #         await manager.shutdown()
    #         Path(db_path).unlink(missing_ok=True)
    #         Path(file_path).unlink(missing_ok=True)

    async def test__get_resource__file_no_modification_no_update(self):
        """Test that get_resource doesn't update content when file hasn't changed."""
        try:
            # Create file
            db_path = create_temp_db_path()
            file_path = create_temp_file("test content")

            manager = ResourceManager(db_path)
            await manager.initialize()
            await manager.add_resource(file_path, chat_pb2.ResourceType.FILE)
            initial_resource = await manager.get_resource(file_path, type=chat_pb2.ResourceType.FILE)  # noqa: E501
            initial_modified = initial_resource.last_modified
            initial_accessed = initial_resource.last_accessed
            await asyncio.sleep(0.1)
            # Get resource again - should only update last_accessed
            new_resource = await manager.get_resource(file_path, type=chat_pb2.ResourceType.FILE)
            assert new_resource.content == "test content"
            assert new_resource.last_modified == initial_modified
            assert new_resource.last_accessed > initial_accessed
        finally:
            await manager.shutdown()
            Path(db_path).unlink(missing_ok=True)
            Path(file_path).unlink(missing_ok=True)

    async def test__get_resource__updates_when_file_modified(self):
        """Test that get_resource returns updated content when file is modified."""
        try:
            db_path = create_temp_db_path()
            file_path = create_temp_file("initial content")

            manager = ResourceManager(db_path)
            await manager.initialize()
            await manager.add_resource(file_path, chat_pb2.ResourceType.FILE)
            initial = await manager.get_resource(file_path, type=chat_pb2.ResourceType.FILE)
            assert initial.content == "initial content"

            # Get again - should see same content and modified time
            # sleep to ensure different last_accessed time
            await asyncio.sleep(0.1)
            same = await manager.get_resource(file_path, type=chat_pb2.ResourceType.FILE)
            assert same.content == "initial content"
            assert same.last_modified == initial.last_modified
            assert same.last_accessed > initial.last_accessed  # This should be updated

            # ensure that last_accessed is updated from `same` to `updated` objects
            await asyncio.sleep(0.1)
            async with aiofiles.open(file_path, 'w') as f:
                await f.write("updated content")

            # Get should see updated content
            updated = await manager.get_resource(file_path, type=chat_pb2.ResourceType.FILE)
            assert updated.content == "updated content"
            assert updated.last_modified > initial.last_modified
            assert updated.last_accessed > same.last_accessed

            # test the underlying record in the database
            async with aiosqlite.connect(db_path) as db:
                cursor = await db.execute(
                    "SELECT content, last_modified, last_accessed FROM resources WHERE path = ?",
                    (file_path,),
                )
                content, last_modified, last_accessed = await cursor.fetchone()
                assert content == "updated content"
                assert last_modified == updated.last_modified
                assert last_accessed == updated.last_accessed
        finally:
            await manager.shutdown()
            Path(db_path).unlink(missing_ok=True)
            Path(file_path).unlink(missing_ok=True)

    async def test__get_resource__file_deleted(self):
        """Test get_resource behavior when file is deleted."""
        try:
            db_path = create_temp_db_path()
            file_path = create_temp_file("test content")

            manager = ResourceManager(db_path)
            await manager.initialize()
            await manager.add_resource(file_path, chat_pb2.ResourceType.FILE)
            await asyncio.sleep(0.1)
            Path(file_path).unlink()
            with pytest.raises(ResourceNotFoundError):
                await manager.get_resource(file_path, type=chat_pb2.ResourceType.FILE)
        finally:
            await manager.shutdown()
            Path(db_path).unlink(missing_ok=True)

    async def test__add_resource__pdf_file__success(self):
        """Test successfully adding a PDF file resource."""
        try:
            db_path = create_temp_db_path()
            pdf_path = 'tests/test_files/pdf/attention_is_all_you_need_short.pdf'
            # ensure file exists
            assert os.path.exists(pdf_path)
            manager = ResourceManager(db_path)
            await manager.initialize()
            await manager.add_resource(
                path=pdf_path,
                type=chat_pb2.ResourceType.FILE,
                metadata={'type': 'pdf'},
            )
            resource = await manager.get_resource(pdf_path, type=chat_pb2.ResourceType.FILE)
            assert resource
            assert resource.path == pdf_path
            assert resource.type == chat_pb2.ResourceType.FILE
            assert resource.content is not None
            assert 'The dominant sequence transduction model' in resource.content
            assert resource.last_accessed
            assert resource.last_modified
            assert resource.metadata['type'] == 'pdf'

            # Verify in database
            async with aiosqlite.connect(db_path) as db:
                cursor = await db.execute(
                    "SELECT type, content FROM resources WHERE path = ?",
                    (pdf_path,),
                )
                result = await cursor.fetchone()
                assert result is not None
                assert result[0] == chat_pb2.ResourceType.FILE
                assert 'The dominant sequence transduction model' in result[1]
        finally:
            await manager.shutdown()
            Path(db_path).unlink(missing_ok=True)

    async def test__get_resource__pdf_file__updates_when_modified(self):
        """Test that get_resource updates when PDF file is modified."""
        db_path = create_temp_db_path()
        try:
            pdf_path = 'tests/test_files/pdf/attention_is_all_you_need_short.pdf'

            manager = ResourceManager(db_path)
            await manager.initialize()
            await manager.add_resource(pdf_path, chat_pb2.ResourceType.FILE)

            initial = await manager.get_resource(pdf_path, type=chat_pb2.ResourceType.FILE)
            assert 'The dominant sequence transduction model' in initial.content
            initial_modified = initial.last_modified
            initial_accessed = initial.last_accessed

            # Get again without modifications - should only update last_accessed
            await asyncio.sleep(0.1)
            same = await manager.get_resource(pdf_path, type=chat_pb2.ResourceType.FILE)
            assert same.content == initial.content
            assert same.last_modified == initial_modified
            assert same.last_accessed > initial_accessed

            # Touch the file to update modified time
            await asyncio.sleep(0.1)
            Path(pdf_path).touch()

            # Get should detect modification and re-extract content
            updated = await manager.get_resource(pdf_path, type=chat_pb2.ResourceType.FILE)
            assert 'The dominant sequence transduction model' in updated.content
            assert updated.last_modified > initial_modified
            assert updated.last_accessed > same.last_accessed

        finally:
            await manager.shutdown()
            Path(db_path).unlink(missing_ok=True)

    async def test__add_resource__ipynb_file__success(self):
        """Test successfully adding a Jupyter Notebook file resource."""
        try:
            db_path = create_temp_db_path()
            ipynb_path = 'tests/test_files/notebooks/simple_notebook.ipynb'
            # ensure file exists
            assert os.path.exists(ipynb_path)
            manager = ResourceManager(db_path)
            await manager.initialize()
            await manager.add_resource(
                path=ipynb_path,
                type=chat_pb2.ResourceType.FILE,
                metadata={'type': 'ipynb'},
            )
            resource = await manager.get_resource(ipynb_path, type=chat_pb2.ResourceType.FILE)
            assert resource
            assert resource.path == ipynb_path
            assert resource.type == chat_pb2.ResourceType.FILE
            assert resource.content is not None
            # see test__extract_jupyter_notebook_content__simple for related test
            assert '[MARKDOWN CELL]' in resource.content
            assert '[CODE CELL]' in resource.content
            assert '[CODE CELL OUTPUT]' in resource.content
            assert "This is a fake notebook." in resource.content
            assert resource.last_accessed
            assert resource.last_modified
            assert resource.metadata['type'] == 'ipynb'

            # Verify in database
            async with aiosqlite.connect(db_path) as db:
                cursor = await db.execute(
                    "SELECT type, content FROM resources WHERE path = ?",
                    (ipynb_path,),
                )
                result = await cursor.fetchone()
                assert result is not None
                assert result[0] == chat_pb2.ResourceType.FILE
                assert 'This is a fake notebook.' in result[1]
        finally:
            await manager.shutdown()
            Path(db_path).unlink(missing_ok=True)


@pytest.mark.asyncio
class TestDirectoryResources:
    """Tests for directory resource handling."""

    async def test__get_resource__directory__simple(self):
        """Test getting a simple directory resource."""
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "file1.txt").touch()
            Path(temp_dir, "file2.txt").touch()

            manager = ResourceManager(create_temp_db_path())
            await manager.initialize()
            resource = await manager.get_resource(temp_dir, type=chat_pb2.ResourceType.DIRECTORY)
            assert resource.path == temp_dir
            assert resource.type == chat_pb2.ResourceType.DIRECTORY
            expected = dedent(f"""
                {os.path.basename(temp_dir)}
                ├── file1.txt
                └── file2.txt""").strip()
            assert resource.content == expected
            assert resource.last_accessed
            assert resource.last_modified

    async def test__get_resource__directory__not_found(self):
        """Test getting a non-existent directory."""
        manager = ResourceManager(create_temp_db_path())
        await manager.initialize()
        with pytest.raises(ResourceNotFoundError):
            await manager.get_resource("/nonexistent/dir", type=chat_pb2.ResourceType.DIRECTORY)

    async def test__get_resource__directory__with_gitignore(self):
        """Test getting a directory with .gitignore."""
        with tempfile.TemporaryDirectory() as temp_dir:
            async with aiofiles.open(Path(temp_dir, ".gitignore"), "w") as f:
                await f.write("*.log")

            Path(temp_dir, "file.txt").touch()
            Path(temp_dir, "ignore.log").touch()

            manager = ResourceManager(create_temp_db_path())
            await manager.initialize()

            resource = await manager.get_resource(temp_dir, type=chat_pb2.ResourceType.DIRECTORY)
            expected = dedent(f"""
                {os.path.basename(temp_dir)}
                └── file.txt""").strip()
            assert resource.content == expected

    async def test__get_resource__directory__updates_on_changes(self):
        """Test directory content updates when files change."""
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "initial.txt").touch()

            manager = ResourceManager(create_temp_db_path())
            await manager.initialize()

            # Get initial state
            resource1 = await manager.get_resource(temp_dir, type=chat_pb2.ResourceType.DIRECTORY)

            # Add new file
            await asyncio.sleep(0.1)  # Ensure different timestamp
            Path(temp_dir, "new.txt").touch()

            # Get updated state
            resource2 = await manager.get_resource(temp_dir, type=chat_pb2.ResourceType.DIRECTORY)

            assert "new.txt" in resource2.content
            assert resource2.last_accessed > resource1.last_accessed


@pytest.mark.asyncio
class TestWebpageResources:
    """Tests for webpage resource handling."""

    async def test__add_resource__webpage__success(self):
        """Test successfully adding a webpage resource."""
        db_path = create_temp_db_path()
        try:
            url = "https://example.com"
            manager = ResourceManager(db_path)
            await manager.initialize()
            await manager.add_resource(
                path=url,
                type=chat_pb2.ResourceType.WEBPAGE,
                metadata={'type': 'webpage'},
            )
            resource = await manager.get_resource(url, type=chat_pb2.ResourceType.WEBPAGE)
            assert resource
            assert resource.path == url
            assert resource.type == chat_pb2.ResourceType.WEBPAGE
            assert resource.content
            assert "Example Domain" in resource.content
            assert resource.last_accessed
            assert resource.last_modified
            assert 'type' in resource.metadata
            assert resource.metadata['type'] == 'webpage'

            # Verify in database
            async with aiosqlite.connect(db_path) as db:
                cursor = await db.execute(
                    "SELECT type, content FROM resources WHERE path = ?",
                    (url,),
                )
                result = await cursor.fetchone()
                assert result is not None
                assert result[0] == chat_pb2.ResourceType.WEBPAGE
                assert "Example Domain" in result[1]
        finally:
            await manager.shutdown()
            Path(db_path).unlink(missing_ok=True)

    async def test__add_resource__webpage__invalid_url(self):
        """Test adding a webpage with an invalid URL."""
        db_path = create_temp_db_path()
        try:
            invalid_url = "not_a_url"
            manager = ResourceManager(db_path)
            await manager.initialize()
            await manager.add_resource(
                path=invalid_url,
                type=chat_pb2.ResourceType.WEBPAGE,
            )
            # Attempt to get should fail since worker couldn't process invalid URL
            with pytest.raises(ResourceNotFoundError):
                await manager.get_resource(invalid_url, type=chat_pb2.ResourceType.WEBPAGE)
        finally:
            await manager.shutdown()
            Path(db_path).unlink(missing_ok=True)

    async def test__concurrent_webpage_fetches(self):
        """Test handling of concurrent webpage resource requests."""
        db_path = create_temp_db_path()
        try:
            url = "https://example.com"
            manager = ResourceManager(db_path, num_workers=2)
            await manager.initialize()

            # Create multiple add tasks for the same webpage
            # This should actually be fairly quick since any requests added while worker is
            # processing/locked get discarded
            tasks = []
            for _ in range(20):
                task = asyncio.create_task(
                    manager.add_resource(url, chat_pb2.ResourceType.WEBPAGE),
                )
                tasks.append(task)
            # Wait for all tasks
            await asyncio.gather(*tasks)

            resource = await manager.get_resource(url, type=chat_pb2.ResourceType.WEBPAGE)
            assert "Example Domain" in resource.content

            # Verify only one entry exists
            async with aiosqlite.connect(db_path) as db:
                cursor = await db.execute(
                    "SELECT COUNT(*) FROM resources WHERE path = ?",
                    (url,),
                )
                count = await cursor.fetchone()
                assert count[0] == 1

        finally:
            await manager.shutdown()
            Path(db_path).unlink(missing_ok=True)

    @SKIP_CI  # arxiv times out in CI
    async def test__add_resource__arxiv_pdf__success(self):
        """Test successfully adding an arXiv PDF as a webpage resource."""
        db_path = create_temp_db_path()
        try:
            url = "https://arxiv.org/pdf/1706.03762"
            manager = ResourceManager(db_path)
            await manager.initialize()
            await manager.add_resource(
                path=url,
                type=chat_pb2.ResourceType.WEBPAGE,
                metadata={'source': 'arxiv'},
            )
            resource = await manager.get_resource(url, type=chat_pb2.ResourceType.WEBPAGE)
            assert resource
            assert resource.path == url
            assert resource.type == chat_pb2.ResourceType.WEBPAGE
            assert resource.content is not None
            # Check for specific content that should be in the Attention paper
            content_lower = resource.content.lower()
            assert "the dominant sequence transduction models are based" in content_lower
            assert "most competitive neural sequence transduction models have an encoder-decoder structure" in content_lower  # noqa: E501
            assert "on the wmt 2014 english-to-german translation task," in content_lower
            assert "in this work, we presented the transformer, the first sequence transduction model" in content_lower  # noqa: E501
            assert resource.last_accessed
            assert resource.last_modified
            assert resource.metadata['source'] == 'arxiv'

            # Verify database content
            async with aiosqlite.connect(db_path) as db:
                cursor = await db.execute(
                    "SELECT type, content FROM resources WHERE path = ?",
                    (url,),
                )
                result = await cursor.fetchone()
                assert result is not None
                assert result[0] == chat_pb2.ResourceType.WEBPAGE
                assert resource.content == result[1]

            # save to diff any future changes.
            async with aiofiles.open('tests/test_files/pdf/attention_is_all_you_need__resource.txt', 'w') as f:  # noqa: E501
                await f.write(resource.content)
        finally:
            await manager.shutdown()
            Path(db_path).unlink(missing_ok=True)

    @SKIP_CI  # arxiv times out in CI
    async def test__concurrent_arxiv_fetches(self):
        """Test handling of concurrent arXiv PDF resource requests."""
        db_path = create_temp_db_path()
        try:
            url = "https://arxiv.org/pdf/1706.03762"
            manager = ResourceManager(db_path, num_workers=2)
            await manager.initialize()

            # Create multiple add tasks for the same PDF
            tasks = []
            for _ in range(10):
                task = asyncio.create_task(
                    manager.add_resource(url, chat_pb2.ResourceType.WEBPAGE),
                )
                tasks.append(task)

            # Wait for all tasks
            await asyncio.gather(*tasks)
            # Verify content is accessible and properly processed
            resource = await manager.get_resource(url, type=chat_pb2.ResourceType.WEBPAGE)
            assert "The dominant sequence transduction model" in resource.content
            assert "Abstract" in resource.content

            # Verify only one entry exists
            async with aiosqlite.connect(db_path) as db:
                cursor = await db.execute(
                    "SELECT COUNT(*) FROM resources WHERE path = ?",
                    (url,),
                )
                count = await cursor.fetchone()
                assert count[0] == 1
        finally:
            await manager.shutdown()
            Path(db_path).unlink(missing_ok=True)

    @SKIP_CI  # arxiv times out in CI
    async def test__invalid_arxiv_pdf_url(self):
        """Test handling of invalid arXiv PDF URLs."""
        db_path = create_temp_db_path()
        try:
            # Use an invalid arXiv PDF URL
            url = "https://arxiv.org/pdf/9999.99999"  # Non-existent paper
            manager = ResourceManager(db_path)
            await manager.initialize()
            await manager.add_resource(url, chat_pb2.ResourceType.WEBPAGE)
            # Attempt to get should fail since worker couldn't fetch the PDF
            with pytest.raises(ResourceNotFoundError):
                await manager.get_resource(url, type=chat_pb2.ResourceType.WEBPAGE)
        finally:
            await manager.shutdown()
            Path(db_path).unlink(missing_ok=True)


@pytest.mark.asyncio
class TestContextCreation:
    """Tests for context creation functionality."""

    async def test__create_context__single_file(self):
        """Test context creation with a single file resource."""
        try:
            db_path = create_temp_db_path()
            file_path = create_temp_file("Test content")

            manager = ResourceManager(db_path)
            await manager.initialize()
            await manager.add_resource(
                path=file_path,
                type=chat_pb2.ResourceType.FILE,
            )
            context, strategies = await manager.create_context(
                [
                    chat_pb2.Resource(
                        path=file_path,
                        type=chat_pb2.ResourceType.FILE,
                    ),
                ],
            )
            assert file_path in context
            assert "Test content" in context
            assert file_path in strategies
            assert strategies[file_path] == ContextType.FULL_TEXT
        finally:
            await manager.shutdown()
            Path(db_path).unlink(missing_ok=True)
            Path(file_path).unlink(missing_ok=True)

    async def test__create_context__multiple_files(self):
        """Test context creation with multiple file resources."""
        try:
            db_path = create_temp_db_path()
            file1_path = create_temp_file("Content 1")
            file2_path = create_temp_file("Content 2")

            manager = ResourceManager(db_path)
            await manager.initialize()
            await manager.add_resource(
                path=file1_path,
                type=chat_pb2.ResourceType.FILE,
            )
            await manager.add_resource(
                path=file2_path,
                type=chat_pb2.ResourceType.FILE,
            )

            context, strategies = await manager.create_context([
                chat_pb2.Resource(
                    path=file1_path,
                    type=chat_pb2.ResourceType.FILE,
                ),
                chat_pb2.Resource(
                    path=file2_path,
                    type=chat_pb2.ResourceType.FILE,
                ),
            ])
            assert file1_path in context
            assert file2_path in context
            assert "Content 1" in context
            assert "Content 2" in context
            assert file1_path in strategies
            assert strategies[file1_path] == ContextType.FULL_TEXT
            assert file2_path in strategies
            assert strategies[file2_path] == ContextType.FULL_TEXT
        finally:
            await manager.shutdown()
            Path(db_path).unlink(missing_ok=True)
            Path(file1_path).unlink(missing_ok=True)
            Path(file2_path).unlink(missing_ok=True)

    async def test__create_context__mixed_resources(self):
        """Test context creation with mixed resource types."""
        try:
            db_path = create_temp_db_path()
            file_path = create_temp_file("File content")
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create files in directory
                Path(temp_dir, "file1.txt").touch()
                Path(temp_dir, "file2.txt").touch()
                manager = ResourceManager(db_path)
                await manager.initialize()
                await manager.add_resource(
                    path=file_path,
                    type=chat_pb2.ResourceType.FILE,
                )

                context, strategies = await manager.create_context([
                    chat_pb2.Resource(
                        path=file_path,
                        type=chat_pb2.ResourceType.FILE,
                    ),
                    chat_pb2.Resource(
                        path=temp_dir,
                        type=chat_pb2.ResourceType.DIRECTORY,
                    ),
                ])
                assert file_path in context
                assert "File content" in context
                assert "file1.txt" in context
                assert "file2.txt" in context
                assert file_path in strategies
                assert strategies[file_path] == ContextType.FULL_TEXT
                assert temp_dir in strategies
                assert strategies[temp_dir] == ContextType.FULL_TEXT
        finally:
            await manager.shutdown()
            Path(db_path).unlink(missing_ok=True)
            Path(file_path).unlink(missing_ok=True)

    async def test__create_context__nonexistent_resource(self):
        """Test context creation with a nonexistent resource."""
        try:
            db_path = create_temp_db_path()
            manager = ResourceManager(db_path)
            await manager.initialize()

            with pytest.raises(ResourceNotFoundError):
                await manager.create_context([
                    chat_pb2.Resource(
                        path="/nonexistent/path",
                        type=chat_pb2.ResourceType.FILE,
                    ),
                ])

        finally:
            await manager.shutdown()
            Path(db_path).unlink(missing_ok=True)

    async def test__create_context__empty_resources(self):
        """Test context creation with empty resource list."""
        try:
            db_path = create_temp_db_path()
            manager = ResourceManager(db_path)
            await manager.initialize()
            context, strategies = await manager.create_context([])
            assert context == ""
            assert not strategies
        finally:
            await manager.shutdown()
            Path(db_path).unlink(missing_ok=True)

    async def test__create_context__webpage_resource(self):
        """Test context creation with a webpage resource."""
        try:
            db_path = create_temp_db_path()
            manager = ResourceManager(db_path)
            await manager.initialize()

            url = "https://example.com"
            await manager.add_resource(
                path=url,
                type=chat_pb2.ResourceType.WEBPAGE,
            )

            context, strategies = await manager.create_context([
                chat_pb2.Resource(
                    path=url,
                    type=chat_pb2.ResourceType.WEBPAGE,
                ),
            ])
            assert url in context
            assert "Example Domain" in context
            assert url in strategies
            assert strategies[url] == ContextType.FULL_TEXT
        finally:
            await manager.shutdown()
            Path(db_path).unlink(missing_ok=True)

    async def test__create_context__duplicate_resources(self):
        """Test context creation with duplicate resources."""
        try:
            db_path = create_temp_db_path()
            file_path = create_temp_file("Test content")

            manager = ResourceManager(db_path)
            await manager.initialize()
            await manager.add_resource(
                path=file_path,
                type=chat_pb2.ResourceType.FILE,
            )

            context, strategies = await manager.create_context([
                chat_pb2.Resource(
                    path=file_path,
                    type=chat_pb2.ResourceType.FILE,
                ),
                chat_pb2.Resource(
                    path=file_path,
                    type=chat_pb2.ResourceType.FILE,
                ),
            ])
            # ensure "test content" only appears once in the context
            assert context.count('Test content') == 1
            assert context.count(file_path) == 1
            assert file_path in strategies
            assert strategies[file_path] == ContextType.FULL_TEXT
        finally:
            await manager.shutdown()
            Path(db_path).unlink(missing_ok=True)
            Path(file_path).unlink(missing_ok=True)


class MockEmbeddingModel:
    """Mock model that returns predictable embeddings."""

    def __init__(self):
        self.encode_count = 0

    def encode(self, texts: str | list[str]) -> np.ndarray | list[np.ndarray]:
        """
        Return embeddings that will give predictable similarities.

        For queries: returns [1, 0, 0]
        For texts containing:
        - 'machine learning': returns vector with 0.9 similarity
        - 'artificial intelligence': returns vector with 0.8 similarity
        - 'computer': returns vector with 0.6 similarity
        - other text: returns vector with 0.2 similarity
        """
        self.encode_count += 1

        if isinstance(texts, str):
            return np.array([1.0, 0.0, 0.0])

        embeddings = []
        for text in texts:
            text = text.lower()  # noqa: PLW2901
            if "machine learning" in text:
                vec = [0.9, 0.436, 0]  # Will give 0.9 similarity
            elif "artificial intelligence" in text:
                vec = [0.8, 0.6, 0]    # Will give 0.8 similarity
            elif "computer" in text:
                vec = [0.6, 0.8, 0]    # Will give 0.6 similarity
            else:
                vec = [0.2, 0.98, 0]   # Will give 0.2 similarity
            embeddings.append(np.array(vec))
        return np.array(embeddings)


@pytest.mark.asyncio
class TestResourceManagerContextRAG:
    """Tests for ResourceManager RAG functionality."""

    async def test_rag_threshold_behavior(self):
        """Test that RAG is used or bypassed based on content length threshold."""
        try:
            db_path = create_temp_db_path()
            # Create test content that will have predictable similarity scores
            content = dedent("""
                01 Won't match but will be included.
                02 This text talks about machine learning concepts.
                03 Won't match but will be included.
                04 Won't match and will not be included.
                05 Won't match but will be included.
                06 Here is text about artificial intelligence.
                07 Here is text about artificial intelligence - it should not be duplicated.
                08 Won't match but will be included.
                09 Some text about computers and programming.
                10 This is other content that won't match well.
            """).strip()
            # Test with content length below threshold
            file_small = create_temp_file(content)
            manager = ResourceManager(
                db_path=db_path,
                rag_scorer=SimilarityScorer(MockEmbeddingModel(), chunk_size=100),
                rag_char_threshold=1000,  # Set high threshold
            )
            await manager.initialize()
            await manager.add_resource(file_small, chat_pb2.ResourceType.FILE)

            context_small, strategies = await manager.create_context(
                [chat_pb2.Resource(path=file_small, type=chat_pb2.ResourceType.FILE)],
                query="machine learning",
                rag_similarity_threshold=0.7,
                context_strategy=ContextStrategy.RAG,
            )
            # Should include full content since below threshold
            assert file_small in context_small
            assert content in context_small
            assert file_small in strategies
            assert strategies[file_small] == ContextType.FULL_TEXT

            context_small, _ = await manager.create_context(
                [chat_pb2.Resource(path=file_small, type=chat_pb2.ResourceType.FILE)],
                query="machine learning",
                rag_similarity_threshold=0.7,
                max_content_length=20,
                context_strategy=ContextStrategy.RAG,
            )
            assert len(context_small) == 20
            assert 'well.' in context_small

            # Test with content length above threshold
            file_large = create_temp_file(content)  # Make content longer
            manager = ResourceManager(
                db_path=db_path,
                # low chunk size ensure each sentence is a chunk
                rag_scorer=SimilarityScorer(MockEmbeddingModel(), chunk_size=10),
                rag_char_threshold=10,  # Set low threshold
            )
            await manager.initialize()
            await manager.add_resource(file_large, chat_pb2.ResourceType.FILE)
            context_large, strategies = await manager.create_context(
                [chat_pb2.Resource(path=file_large, type=chat_pb2.ResourceType.FILE)],
                query="machine learning",
                rag_similarity_threshold=0.7,
                context_strategy=ContextStrategy.RAG,
            )
            assert file_large in context_large
            assert file_large in strategies
            assert strategies[file_large] == ContextType.RAG
            # machine learning will be above threshold
            assert context_large.count("01") == 1
            assert context_large.count("02") == 1
            assert context_large.count("03") == 1
            assert "04" not in context_large
            assert context_large.count("05") == 1
            assert context_large.count("06") == 1
            assert context_large.count("07") == 1
            assert context_large.count("08") == 1
            assert "09" not in context_large
            assert "10" not in context_large

            context_large, _ = await manager.create_context(
                [chat_pb2.Resource(path=file_large, type=chat_pb2.ResourceType.FILE)],
                query="machine learning",
                rag_similarity_threshold=0.85,
                context_strategy=ContextStrategy.RAG,
            )
            # machine learning will be above threshold
            assert context_large.count("01") == 1
            assert context_large.count("02") == 1
            assert context_large.count("03") == 1
            assert "04" not in context_large
            assert "05" not in context_large
            assert "06" not in context_large
            assert "07" not in context_large
            assert "08" not in context_large
            assert "09" not in context_large
            assert "10" not in context_large
        finally:
            await manager.shutdown()
            Path(db_path).unlink(missing_ok=True)
            Path(file_small).unlink(missing_ok=True)
            Path(file_large).unlink(missing_ok=True)

    async def test_rag_with_code_files(self):
        """Test that code files bypass RAG regardless of size or threshold."""
        try:
            db_path = create_temp_db_path()
            content = "def machine_learning(): pass\n" * 100  # Make it about ML but in code
            code_file = create_temp_file(content)
            code_path = f"{code_file}.py"
            os.rename(code_file, code_path)
            manager = ResourceManager(
                db_path=db_path,
                rag_scorer=SimilarityScorer(MockEmbeddingModel(), chunk_size=10),
                # Set low threshold to ensure we don't create a false negative by bypassing RAG
                rag_char_threshold=10,
            )
            await manager.initialize()
            await manager.add_resource(code_path, chat_pb2.ResourceType.FILE)
            context, strategies = await manager.create_context(
                [chat_pb2.Resource(path=code_path, type=chat_pb2.ResourceType.FILE)],
                query="machine learning",
                rag_similarity_threshold=0.7,
                context_strategy=ContextStrategy.RAG,
            )
            # Should include full content, not RAG results
            assert code_path in context
            assert context.count("def machine_learning()") == 100
            assert code_path in strategies
            assert strategies[code_path] == ContextType.FULL_TEXT
        finally:
            await manager.shutdown()
            Path(db_path).unlink(missing_ok=True)
            Path(code_path).unlink(missing_ok=True)

    async def test_no_relevant_chunks_found(self):
        """Test handling when no chunks meet similarity threshold."""
        try:
            db_path = create_temp_db_path()
            content = "Unrelated content about various topics.\n" * 50
            file_path = create_temp_file(content)
            manager = ResourceManager(
                db_path=db_path,
                rag_scorer=SimilarityScorer(MockEmbeddingModel(), chunk_size=10),
                rag_char_threshold=100,
            )
            await manager.initialize()
            await manager.add_resource(file_path, chat_pb2.ResourceType.FILE)
            context, strategies = await manager.create_context(
                [chat_pb2.Resource(path=file_path, type=chat_pb2.ResourceType.FILE)],
                query="machine learning",
                rag_similarity_threshold=0.7,
                context_strategy=ContextStrategy.RAG,
            )
            # Document should not appear in context at all since no chunks met threshold
            assert context == ''
            assert file_path in strategies
            assert strategies[file_path] == ContextType.RAG
        finally:
            await manager.shutdown()
            Path(db_path).unlink(missing_ok=True)
            Path(file_path).unlink(missing_ok=True)

    async def test_mixed_file_types(self):
        """Test handling of mixed file types with RAG."""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                db_path = create_temp_db_path()
                # Create various file types
                text_content = "Machine learning concepts.\n" * 50
                text_file = create_temp_file(text_content)
                code_content = "def machine_learning(): pass\n" * 50
                code_file = create_temp_file(code_content)
                code_path = f"{code_file}.py"
                os.rename(code_file, code_path)
                # Create files in directory
                Path(temp_dir, "file1.txt").write_text("ML content")
                Path(temp_dir, "file2.txt").write_text("AI content")
                manager = ResourceManager(
                    db_path=db_path,
                    rag_scorer=SimilarityScorer(MockEmbeddingModel(), chunk_size=10),
                    rag_char_threshold=10,
                )
                await manager.initialize()
                await manager.add_resource(text_file, chat_pb2.ResourceType.FILE)
                await manager.add_resource(code_path, chat_pb2.ResourceType.FILE)
                context, strategies = await manager.create_context(
                    [
                        chat_pb2.Resource(path=text_file, type=chat_pb2.ResourceType.FILE),
                        chat_pb2.Resource(path=code_path, type=chat_pb2.ResourceType.FILE),
                        chat_pb2.Resource(path=temp_dir, type=chat_pb2.ResourceType.DIRECTORY),
                    ],
                    query="machine learning",
                    rag_similarity_threshold=0.99,
                    context_strategy=ContextStrategy.RAG,
                )
                # Text file should use RAG and threshold is set to 0.99
                assert context.count("Machine learning concepts") == 0
                # Code file should appear in full
                assert context.count("def machine_learning()") == 50
                # Directory should show tree structure
                assert "file1.txt" in context
                assert "file2.txt" in context
                assert text_file in strategies
                assert strategies[text_file] == ContextType.RAG
                assert code_path in strategies
                assert strategies[code_path] == ContextType.FULL_TEXT
                assert temp_dir in strategies
                assert strategies[temp_dir] == ContextType.FULL_TEXT
            finally:
                await manager.shutdown()
                Path(db_path).unlink(missing_ok=True)
                Path(text_file).unlink(missing_ok=True)
                Path(code_path).unlink(missing_ok=True)

    async def test_rag_threshold_behavior__max_k(self):
        """Test that RAG is used or bypassed based on content length threshold."""
        try:
            db_path = create_temp_db_path()
            # Create test content that will have predictable similarity scores
            content = dedent("""
                01 Won't match but will be included.
                02 This text talks about machine learning concepts.
                03 Won't match but will be included.
                04 Won't match and will not be included.
                05 Won't match but will be included.
                06 Here is text about artificial intelligence.
                07 Here is text about artificial intelligence - it should not be duplicated.
                08 Won't match but will be included.
                09 Some text about computers and programming.
                10 This is other content that won't match well.
            """).strip()
            file_large = create_temp_file(content)  # Make content longer
            manager = ResourceManager(
                db_path=db_path,
                # low chunk size ensure each sentence is a chunk
                rag_scorer=SimilarityScorer(MockEmbeddingModel(), chunk_size=10),
                rag_char_threshold=10,  # Set low threshold
            )
            await manager.initialize()
            await manager.add_resource(file_large, chat_pb2.ResourceType.FILE)

            context_large, strategies = await manager.create_context(
                [chat_pb2.Resource(path=file_large, type=chat_pb2.ResourceType.FILE)],
                query="machine learning",
                rag_similarity_threshold=0.7,
                rag_max_k=3,
                context_strategy=ContextStrategy.RAG,
            )
            assert file_large in context_large
            assert file_large in strategies
            assert strategies[file_large] == ContextType.RAG
            # machine learning will be above threshold
            assert context_large.count("01") == 1
            assert context_large.count("02") == 1
            assert context_large.count("03") == 1
            assert "04" not in context_large
            assert context_large.count("05") == 1
            assert context_large.count("06") == 1
            assert context_large.count("07") == 1
            assert context_large.count("08") == 1
            assert "09" not in context_large
            assert "10" not in context_large

            context, strategies = await manager.create_context(
                [chat_pb2.Resource(path=file_large, type=chat_pb2.ResourceType.FILE)],
                query="machine learning",
                rag_similarity_threshold=0.7,
                rag_max_k=1,
                context_strategy=ContextStrategy.RAG,
            )
            assert file_large in context
            assert file_large in strategies
            assert strategies[file_large] == ContextType.RAG
            # machine learning will be above threshold
            assert context.count("01") == 1
            assert context.count("02") == 1
            assert context.count("03") == 1
            assert "04" not in context
            assert "05" not in context
            assert "06" not in context
            assert "07" not in context
            assert "08" not in context
            assert "09" not in context
            assert "10" not in context

            context, strategies = await manager.create_context(
                [chat_pb2.Resource(path=file_large, type=chat_pb2.ResourceType.FILE)],
                query="machine learning",
                rag_similarity_threshold=0.85,
                rag_max_k=1,
                context_strategy=ContextStrategy.RAG,
            )
            assert file_large in context
            assert file_large in strategies
            assert strategies[file_large] == ContextType.RAG
            # machine learning will be above threshold
            assert context.count("01") == 1
            assert context.count("02") == 1
            assert context.count("03") == 1
            assert "04" not in context
            assert "05" not in context
            assert "06" not in context
            assert "07" not in context
            assert "08" not in context
            assert "09" not in context
            assert "10" not in context
        finally:
            await manager.shutdown()
            Path(db_path).unlink(missing_ok=True)
            Path(file_large).unlink(missing_ok=True)


@pytest.mark.asyncio
class TestResourceManagerContextAuto:
    """Tests the `ResourceStrategy.AUTO` with `create_context`."""

    async def test_auto_strategy_requires_query(self):
        """Test that AUTO strategy requires a query."""
        try:
            db_path = create_temp_db_path()
            file_path = create_temp_file("Test content")
            manager = ResourceManager(
                db_path=db_path,
                rag_scorer=SimilarityScorer(MockEmbeddingModel()),
            )
            await manager.initialize()
            await manager.add_resource(file_path, chat_pb2.ResourceType.FILE)
            with pytest.raises(ValueError):  # noqa: PT011
                await manager.create_context(
                    [chat_pb2.Resource(path=file_path, type=chat_pb2.ResourceType.FILE)],
                    query=None,
                    context_strategy=ContextStrategy.AUTO,
                )
        finally:
            await manager.shutdown()
            Path(db_path).unlink(missing_ok=True)
            Path(file_path).unlink(missing_ok=True)

    async def test_auto_strategy_requires_model(self):
        """Test that AUTO strategy requires a model to be configured."""
        try:
            db_path = create_temp_db_path()
            file_path = create_temp_file("Test content")
            manager = ResourceManager(
                db_path=db_path,
                rag_scorer=SimilarityScorer(MockEmbeddingModel()),
                context_strategy_model_config=None,
            )
            await manager.initialize()
            await manager.add_resource(file_path, chat_pb2.ResourceType.FILE)
            with pytest.raises(ValueError):  # noqa: PT011
                await manager.create_context(
                    [chat_pb2.Resource(path=file_path, type=chat_pb2.ResourceType.FILE)],
                    query="test query",
                    context_strategy=ContextStrategy.AUTO,
                )
        finally:
            await manager.shutdown()
            Path(db_path).unlink(missing_ok=True)
            Path(file_path).unlink(missing_ok=True)

    async def test_auto_strategy_code_files(self):
        """Test that code files always use FULL_TEXT even if agent suggests RAG."""
        try:
            db_path = create_temp_db_path()
            # Create a Python file with ML-related content that might trigger RAG
            code_content = "def train_model():\n    # ML training code\n    pass\n" * 100
            code_file = create_temp_file(code_content)
            code_path = f"{code_file}.py"
            os.rename(code_file, code_path)
            manager = ResourceManager(
                db_path=db_path,
                rag_scorer=SimilarityScorer(MockEmbeddingModel(), chunk_size=10),
                # set low threshold to ensure we don't create a false negative by bypassing RAG
                rag_char_threshold=10,
                context_strategy_model_config={
                    'client_type': 'MockAsyncOpenAIStructuredOutput',
                    'model_name': 'MockModel',
                    'mock_responses': {
                        'parsed': MockContextStrategies(
                            strategies=[
                                MockContextStrategy(
                                    resource_name=code_path,
                                    context_type=ContextType.RAG,
                                    reasoning='Mock reasoning',
                                ),
                            ],
                        ),
                    },
                },
            )
            await manager.initialize()
            await manager.add_resource(code_path, chat_pb2.ResourceType.FILE)
            context, strategies = await manager.create_context(
                [chat_pb2.Resource(path=code_path, type=chat_pb2.ResourceType.FILE)],
                query="How does the model training work?",
                context_strategy=ContextStrategy.AUTO,
            )
            assert code_path in context
            assert context.count("def train_model()") == 100  # Full content included
            assert code_path in strategies
            assert strategies[code_path] == ContextType.FULL_TEXT
        finally:
            await manager.shutdown()
            Path(db_path).unlink(missing_ok=True)
            Path(code_path).unlink(missing_ok=True)

    async def test_auto_strategy_mixed_content_types(self):
        """Test AUTO strategy with mixed content types and queries."""
        try:
            db_path = create_temp_db_path()
            # readme content with length over threshold so we should use RAG
            readme_content = "# Overview\nThe project name is `project_abc`. " * 50
            readme_path = create_temp_file(
                readme_content,
                prefix='project_readme_',
                suffix='.md',
            )
            # code content with length over threshold but we should override RAG and use FULL
            code_content = "def process_data(): pass\n" * 50
            code_path = create_temp_file(
                code_content,
                prefix='server_generated_',
                suffix='.py',
            )

            manager = ResourceManager(
                db_path=db_path,
                rag_scorer=SimilarityScorer(MockEmbeddingModel(), chunk_size=10),
                # set low threshold to ensure we don't create a false negative by bypassing RAG
                rag_char_threshold=10,
                context_strategy_model_config={
                    'client_type': 'MockAsyncOpenAIStructuredOutput',
                    'model_name': 'MockModel',
                    'mock_responses': {
                        'parsed': MockContextStrategies(strategies=[
                            MockContextStrategy(
                                resource_name=readme_path,
                                context_type=ContextType.RAG,
                                reasoning='Mock reasoning',
                            ),
                            MockContextStrategy(
                                resource_name=code_path,
                                context_type=ContextType.IGNORE,
                                reasoning='Mock reasoning',
                            ),
                        ]),
                    },
                },
            )
            await manager.initialize()
            await manager.add_resource(readme_path, chat_pb2.ResourceType.FILE)
            await manager.add_resource(code_path, chat_pb2.ResourceType.FILE)

            # Test with documentation-focused query
            _, strategies = await manager.create_context(
                [
                    chat_pb2.Resource(path=readme_path, type=chat_pb2.ResourceType.FILE),
                    chat_pb2.Resource(path=code_path, type=chat_pb2.ResourceType.FILE),
                ],
                query="What is the project name as defined in the overview of the readme?",
                context_strategy=ContextStrategy.AUTO,
            )

            assert readme_path in strategies
            assert code_path in strategies
            # Documentation should be included
            assert strategies[readme_path] == ContextType.RAG
            # Code should be ignored for this query
            assert strategies[code_path] == ContextType.IGNORE

            # # Test with code-focused query
            # _, strategies = await manager.create_context(
            #     [
            #         chat_pb2.Resource(path=readme_path, type=chat_pb2.ResourceType.FILE),
            #         chat_pb2.Resource(path=code_path, type=chat_pb2.ResourceType.FILE),
            #     ],
            #     query="What is the `extract_data` function in the generate code in the server?",
            #     context_strategy=ContextStrategy.AUTO,
            # )
            # # Code should be included and use FULL_TEXT
            # assert strategies[code_path] == ContextType.FULL_TEXT
            # # Documentation might be ignored or RAG
            # assert strategies[readme_path] in [ContextType.IGNORE, ContextType.RAG]
        finally:
            await manager.shutdown()
            Path(db_path).unlink(missing_ok=True)
            Path(readme_path).unlink(missing_ok=True)
            Path(code_path).unlink(missing_ok=True)
