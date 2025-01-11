"""Fixtures and helper functions for testing."""
import os
import tempfile
import pytest


SKIP_CI = pytest.mark.skipif(
    os.getenv('SKIP_CI_TESTS') == 'true',
    reason='Skipping test in CI environment',
)

def create_temp_file(content: str, prefix: str | None = None, suffix: str | None = None) -> str:
    """Create a temporary file with content."""
    f = tempfile.NamedTemporaryFile(  # noqa: SIM115
        mode='w',
        delete=False,
        prefix=prefix,
        suffix=suffix,
    )
    f.write(content)
    f.close()
    return f.name
