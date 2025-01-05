"""Fixtures and helper functions for testing."""
import tempfile


def create_temp_file(content: str) -> str:
    """Create a temporary file with content."""
    f = tempfile.NamedTemporaryFile(mode='w', delete=False)  # noqa: SIM115
    f.write(content)
    f.close()
    return f.name
