"""Test cases for the async_merge module."""
from dataclasses import dataclass
import time
import pytest
import asyncio

from server.async_merge import AsyncMerge

async def simple_generator(items, delay=0):  # noqa: ANN001
    """Helper to create an async generator with optional delay."""
    for item in items:
        if delay:
            await asyncio.sleep(delay)
        yield item


async def failing_generator():
    """Helper that raises an exception."""
    await asyncio.sleep(0.1)
    raise ValueError("Generator failed")
    yield 1  # need this to make it a generator


@pytest.mark.asyncio
async def test__async_merge__basic():
    """Test basic merging of two simple generators."""
    gen1 = simple_generator([1, 2, 3])
    gen2 = simple_generator([4, 5, 6])

    results = []
    async for item in AsyncMerge([gen1, gen2]):
        results.append(item)

    assert len(results) == 6
    # Note: Order between generators not guaranteed
    assert sorted(results) == [1, 2, 3, 4, 5, 6]

@pytest.mark.asyncio
async def test__async_merge__different_lengths():
    """Test merging generators of different lengths."""
    gen1 = simple_generator([1])
    gen2 = simple_generator([2, 3, 4])

    results = []
    async for item in AsyncMerge([gen1, gen2]):
        results.append(item)

    assert len(results) == 4
    assert sorted(results) == [1, 2, 3, 4]

@pytest.mark.asyncio
async def test__async_merge__with_delays():
    """Test merging generators with different timing."""
    gen1 = simple_generator([1, 2], delay=0.1)  # Slower generator
    gen2 = simple_generator([3, 4], delay=0.05)  # Faster generator

    results = []
    async for item in AsyncMerge([gen1, gen2]):
        results.append(item)
    assert len(results) == 4
    # First item from faster generator (3) should be first or second
    assert 3 in results[:2]

@pytest.mark.asyncio
async def test__async_merge__error_handling():
    """Test that errors in generators are properly propagated."""
    gen1 = simple_generator([1, 2])
    gen2 = failing_generator()

    results = []
    with pytest.raises(ValueError, match="Generator failed"):  # noqa: PT012
        async for item in AsyncMerge([gen1, gen2]):
            results.append(item)
    # Should have received some items before the error
    assert len(results) > 0

@pytest.mark.asyncio
async def test__async_merge__empty_generators():
    """Test merging empty generators."""
    gen1 = simple_generator([])
    gen2 = simple_generator([])

    results = []
    async for item in AsyncMerge([gen1, gen2]):
        results.append(item)

    assert len(results) == 0

@pytest.mark.asyncio
async def test__async_merge__single_generator():
    """Test merging a single generator."""
    gen = simple_generator([1, 2, 3])

    results = []
    async for item in AsyncMerge([gen]):
        results.append(item)

    assert results == [1, 2, 3]

@pytest.mark.asyncio
async def test__async_merge__cleanup():
    """Test that resources are cleaned up properly."""
    cleanup_called = False

    async def generator_with_cleanup():  # noqa: ANN202
        try:
            yield 1
            yield 2
        finally:
            nonlocal cleanup_called
            cleanup_called = True

    gen = generator_with_cleanup()
    async for _ in AsyncMerge([gen]):
        pass

    # Allow cleanup to complete
    await asyncio.sleep(0)
    assert cleanup_called

@dataclass
class MockResponse:  # noqa: D101
    model_index: int
    content: str
    type: str  # 'chunk' or 'summary'

@pytest.mark.asyncio
async def test__async_merge__realistic_payloads():
    async def mock_model_stream(index, num_chunks):  # noqa: ANN001, ANN202
        for i in range(num_chunks):
            await asyncio.sleep(0.01)  # Variable delays
            yield MockResponse(index, f"chunk {i}", "chunk")
        yield MockResponse(index, "summary", "summary")

    num_chunks_0 = 20
    num_chunks_1 = 5

    streams = [
        mock_model_stream(0, num_chunks_0),  # More chunks
        mock_model_stream(1, num_chunks_1),   # Fewer chunks
    ]

    chunks = []
    summaries = []
    async for response in AsyncMerge(streams):
        if response.type == "chunk":
            chunks.append(response)
        else:
            summaries.append(response)


    assert len(summaries) == 2

    # verify that the chunks are interleaved; it's not likely that all chunks from one model will
    # be processed before the other; that would indicate we are processing sequentially
    def are_interleaved(values: list):  # noqa: ANN202
        # Find the indices where the value changes (transitions)
        transitions = sum(values[i] != values[i + 1] for i in range(len(values) - 1))
        return transitions > 1
    assert are_interleaved([0, 1, 0, 1, 0, 1])
    assert are_interleaved([0, 1, 1, 1, 0, 1])
    assert are_interleaved([0, 1, 1, 1, 1, 0])
    assert not are_interleaved([0, 0, 0, 0, 0, 0])
    assert not are_interleaved([1, 1, 1, 1, 1, 1])
    assert not are_interleaved([0, 0, 0, 1, 1, 1])
    assert not are_interleaved([1, 1, 1, 0, 0, 0])

    # now check chunk indexes; it would be extremely unlikely that all chunks from one model
    # would be processed before the other
    assert are_interleaved([c.model_index for c in chunks])

    # Verify ordering within each model stream
    model0_chunks = [r for r in chunks if r.model_index == 0]
    model1_chunks = [r for r in chunks if r.model_index == 1]

    assert len(model0_chunks) == num_chunks_0
    assert len(model1_chunks) == num_chunks_1

    assert all(c.model_index == 0 for c in model0_chunks)
    assert all(c.model_index == 1 for c in model1_chunks)
    assert [c.content for c in model0_chunks] == [f"chunk {i}" for i in range(num_chunks_0)]
    assert [c.content for c in model1_chunks] == [f"chunk {i}" for i in range(num_chunks_1)]


@pytest.mark.asyncio
async def test__async_merge__concurrent_items():
    """
    Test that AsyncMerge properly handles items arriving concurrently and tests that generators
    are interleaving rather than processed sequentially.
    """
    num_samples = 1000
    num_generators = 5
    import random
    async def generator(_id: int):  # noqa: ANN202
        for i in range(num_samples):
            await asyncio.sleep(random.uniform(0, 0.001))
            yield f"gen{_id}-{i}"

    generators = [generator(i) for i in range(num_generators)]
    merger = AsyncMerge(generators)
    results = []
    async for item in merger:
        await asyncio.sleep(0.0001)  # Processing delay
        results.append(item)

    # Original verification
    expected_total = num_samples * num_generators
    assert len(results) == expected_total, f"Expected {expected_total} items but got {len(results)}"  # noqa: E501

    for i in range(num_generators):
        gen_items = [r for r in results if r.startswith(f"gen{i}-")]
        assert len(gen_items) == num_samples, f"Generator {i} missing items. Expected {num_samples} but got {len(gen_items)}"  # noqa: E501

    for i in range(num_generators):
        for j in range(num_samples):
            assert f"gen{i}-{j}" in results, f"Missing item gen{i}-{j}"

    # Add interleaving verification
    # Check first 100 items to verify we're getting items from different generators
    sample_slice = results[:100]
    generators_seen = set()
    for item in sample_slice:
        gen_id = int(item.split('-')[0].replace('gen', ''))
        generators_seen.add(gen_id)

    # We should see items from multiple generators in the first 100 items
    assert len(generators_seen) > 1, "No interleaving detected - only seeing items from one generator"  # noqa: E501

    # More stringent test: no more than N consecutive items from same generator
    max_consecutive = num_samples * 0.05  # Allow some bursts but not too many
    current_gen = None
    consecutive_count = 0

    for item in results:
        gen_id = int(item.split('-')[0].replace('gen', ''))
        if gen_id == current_gen:
            consecutive_count += 1
        else:
            current_gen = gen_id
            consecutive_count = 1
        assert consecutive_count <= max_consecutive, f"Too many consecutive items from generator {gen_id}"  # noqa: E501


@pytest.mark.asyncio
async def test__async_merge__fast_not_delayed_by_slow():
    """Test that slow generators don't delay fast generators."""
    async def fast_generator():  # noqa: ANN202
        for i in range(100):
            yield f"fast-{i}"

    async def slow_generator():  # noqa: ANN202
        for i in range(5):
            yield f"slow-{i}"
            await asyncio.sleep(0.1)  # slow

    merger = AsyncMerge([slow_generator(), fast_generator()])
    # Track timing of fast items
    fast_times = []
    prev_fast_time = None
    results = []
    async for item in merger:
        # print(f"{time.strftime('%H:%M:%S')} - {item}")
        results.append(item)
        if item.startswith("fast"):
            current_time = time.time()
            if prev_fast_time is not None:
                fast_times.append(current_time - prev_fast_time)
            prev_fast_time = current_time

    assert results[0] == "slow-0"
    assert results[1] == "fast-0"
    assert results[2] == "fast-1"
    assert results[-1] == "slow-4"

    # Calculate average time between fast items
    avg_fast_time = sum(fast_times) / len(fast_times)
    # The average time between fast items should be very small
    # (much smaller than the slow generator's sleep time)
    assert avg_fast_time < 0.01, f"Fast items taking too long ({avg_fast_time:.3f}s average)"


@pytest.mark.asyncio
async def test__async_merge__verify_timing():
    """
    Verify that AsyncMerge maintains the timing intervals of the source generators.

    Creates a fast generator (0.1s intervals) and slow generator (0.3s intervals)
    and verifies that the merged stream preserves these intervals without bunching
    items together. For example, if fast items are bunched, we might see multiple
    items arrive with near-zero intervals instead of the expected 0.1s gaps.
    """
    async def fast_generator():  # noqa: ANN202
        for i in range(5):
            yield f"fast-{i}"
            await asyncio.sleep(0.1)  # Should see items every 0.1s

    async def slow_generator():  # noqa: ANN202
        for i in range(2):
            yield f"slow-{i}"
            await asyncio.sleep(0.3)  # Should see items every 0.3s

    merger = AsyncMerge([slow_generator(), fast_generator()])
    results = []
    timestamps = []
    async for item in merger:
        results.append(item)
        timestamps.append((item, time.time()))

    # Verify intervals between fast items are ~0.1s
    fast_times = [(item, t) for item, t in timestamps if item.startswith("fast")]
    fast_intervals = [
        fast_times[i][1] - fast_times[i-1][1]
        for i in range(1, len(fast_times))
    ]
    # Allow some wiggle room in timing but ensure no bunching
    for interval in fast_intervals:
        assert 0.08 < interval < 0.15, \
            f"Fast items should be ~0.1s apart, got {interval:.3f}s"
    # Verify intervals between slow items are ~0.3s
    slow_times = [(item, t) for item, t in timestamps if item.startswith("slow")]
    slow_intervals = [
        slow_times[i][1] - slow_times[i-1][1]
        for i in range(1, len(slow_times))
    ]
    for interval in slow_intervals:
        assert 0.25 < interval < 0.35, \
            f"Slow items should be ~0.3s apart, got {interval:.3f}s"
