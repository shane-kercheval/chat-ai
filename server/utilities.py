"""Utilities for working with PDFs."""
from enum import Enum
import os
from pathlib import Path
import re
from typing import TypeVar
import aiohttp
import aiofiles
import pathspec
from pypdf import PdfReader
import contextlib
from bs4 import BeautifulSoup

CODE_EXTENSIONS = {
    '.py', '.js', '.ts', '.html', '.css', '.cpp', '.c', '.h', '.hpp', '.cs',
    '.java', '.rs', '.go', '.swift', '.kt', '.rb', '.php', '.scala',
    '.sql', '.sh', '.bash', '.json', '.yaml', '.yml', '.toml', '.jsx',
    '.tsx', '.vue', '.dart', '.r', '.m', '.f90', '.f95', '.f03', '.xml',
    '.jsp', '.asp', '.aspx', '.bat', '.ps1', '.ini', '.cfg', '.env',
    '.conf', '.csx', '.gradle', '.properties', '.ipynb', '.pyo', '.pyc',
    '.erb', '.zsh', '.ksh', '.cmake', '.pl', '.perl', '.lua', '.groovy',
    '.coffee', '.svelte', '.astro', '.rmd', '.lock',
}


async def extract_text_from_pdf(pdf_path: str, delete_afterwards: bool = False) -> str:
    """
    Asynchronously extract text from a PDF.

    Args:
        pdf_path: The path to the PDF.
        delete_afterwards: Whether to delete the PDF after extracting the text.
    """
    is_url = pdf_path.startswith(('http:', 'https:')) or 'www.' in pdf_path
    temp_file_path = None
    try:
        # If it's a URL, download the PDF to a temporary file
        if is_url:
            async with aiohttp.ClientSession() as session, session.get(pdf_path) as response:
                response.raise_for_status()
                content = await response.read()
                # Use aiofiles for async file writing
                async with aiofiles.tempfile.NamedTemporaryFile(
                    mode='wb',
                    suffix='.pdf',
                    delete=False,
                ) as temp_file:
                    await temp_file.write(content)
                    temp_file_path = temp_file.name
        else:
            temp_file_path = pdf_path
        # Extract text from the PDF (this part remains synchronous)
        reader = PdfReader(temp_file_path)
        return '\n\n'.join(page.extract_text() for page in reader.pages).strip()
    finally:
        if (is_url or delete_afterwards) and temp_file_path:
            with contextlib.suppress(OSError):
                os.remove(temp_file_path)


def clean_text_from_pdf(
        text: str,
        include_at: str | None = None,
        exclude_at: str | None = None,
    ) -> str:
    """
    Clean text from a PDF.

    Args:
        text: The text to clean.
        include_at: Include text starting from this string.
        exclude_at: Exclude text starting from this string.
    """
    text = text.strip()
    if include_at:
        index = text.find(include_at)
        assert index >= 0
        text = text[index:]
    if exclude_at:
        index = text.find(exclude_at)
        assert index >= 0
        text = text[:index]

    # More sophisticated references section detection
    # Split text into lines
    lines = text.split('\n')

    # Patterns to detect references section start
    references_patterns = [
        r'^\s*(References|Bibliography|Works\s*Cited)\s*:?$',  # Exact match with optional colon
        r'^\s*(\d+\.\s*)?(References|Bibliography|Works\s*Cited)\s*:?$',  # Numbered section
    ]
    # Find the index of the references section
    references_index = -1
    for i, line in enumerate(lines):
        if any(re.match(pattern, line.strip(), re.IGNORECASE) for pattern in references_patterns):
            references_index = i
            break
    # If references section found, truncate the text
    if references_index != -1:
        lines = lines[:references_index]

    text = '\n'.join(lines)
    # Fix arbitrary newlines
    # Replace newlines not preceded by a sentence-ending punctuation with a space
    # This pattern matches if a newline is not preceded by ., !, ?, or ] (assuming references might
    # end a sentence)
    text = re.sub(r'\n(?=[a-z\[\()])', ' ', text)
    # Remove extra spaces and tabs but preserve newlines
    text = re.sub(r'[^\S\n]+', ' ', text)
    text = text.replace('<EOS>', '\n')
    text = text.replace('<eos>', '\n')
    text = text.replace('\n\n', '\n')
    text = text.replace('<pad>', '')
    footnote_symbols = ['†', '‡', '¶', '§', '‖', '※']
    lines = text.split('\n')

    processed_lines = []
    for line in lines:
        line = line.strip()  # noqa: PLW2901
        # Remove lines that start with a footnote symbol
        if any(line.startswith(symbol) for symbol in footnote_symbols) \
                or line.isdigit():
                # or re.match(r'^Figure \d+:', line):
            continue
        if not line:
            continue
        processed_lines.append(line)
    return '\n\n'.join(processed_lines).strip()


class GitIgnoreMatcher:
    """Handles gitignore pattern matching with support for negation patterns."""

    def __init__(self, patterns: list[str]):
        self.negation_patterns = [p[1:] for p in patterns if p.startswith('!')]
        self.ignore_patterns = [p for p in patterns if not p.startswith('!')]
        self.negations = (
            pathspec.PathSpec.from_lines('gitwildmatch', self.negation_patterns)
            if self.negation_patterns else None
        )
        self.ignores = (
            pathspec.PathSpec.from_lines('gitwildmatch', self.ignore_patterns)
            if self.ignore_patterns else None
        )

    def should_include(self, path: str) -> bool:
        """
        Determine if a path should be included based on gitignore rules.

        Args:
            path: The path to check, relative to the gitignore file location

        Returns:
            True if the path should be included, False if it should be ignored
        """
        # Check negation patterns first
        if self.negations:
            # Include if path matches negation pattern
            if self.negations.match_file(path):
                return True
            # Include if path is parent of negated pattern
            for pattern in self.negation_patterns:
                if pattern.startswith(path + '/'):
                    return True
        # If not explicitly negated, include only if not ignored
        return not self.ignores or not self.ignores.match_file(path)


async def _read_gitignore(directory: str) -> pathspec.PathSpec | None:
    """Read .gitignore file and return a PathSpec object."""
    gitignore_path = os.path.join(directory, '.gitignore')
    try:
        async with aiofiles.open(gitignore_path) as f:
            patterns = []
            for line in (await f.read()).splitlines():
                line = line.strip()  # noqa: PLW2901
                if line and not line.startswith('#'):
                    # Remove leading slashes as pathspec handles paths relative to gitignore
                    # location
                    while line.startswith('/'):
                        line = line[1:]  # noqa: PLW2901
                    patterns.append(line)
            return pathspec.PathSpec.from_lines('gitwildmatch', patterns)
    except FileNotFoundError:
        return None


async def generate_directory_tree(path: str) -> str:
    """Generate a tree structure string of the directory."""
    if not os.path.exists(path):
        raise ValueError(f"Path is not a directory: {path}")
    if not os.path.isdir(path):
        raise ValueError(f"Path is not a directory: {path}")

    output = []
    prefix = "├── "
    prefix_last = "└── "
    prefix_empty = "│   "
    prefix_nothing = "    "

    async def _inner_generate(
            current_path: Path,
            prefix_parts: list[str],
            parent_gitignore: pathspec.PathSpec | None,
            root_path: Path,
        ) -> None:
        # Check for local .gitignore and combine with parent patterns
        local_gitignore = await _read_gitignore(str(current_path))
        current_gitignore = parent_gitignore

        if local_gitignore:
            if current_gitignore:
                # Combine patterns from both gitignores
                all_patterns = current_gitignore.patterns + local_gitignore.patterns
                current_gitignore = pathspec.PathSpec(all_patterns)
            else:
                current_gitignore = local_gitignore

        # Get entries and filter hidden files
        entries = sorted(
            (e for e in os.scandir(current_path) if not e.name.startswith('.')),
            key=lambda e: e.name,
        )

        # Filter gitignored files if we have a gitignore spec
        if current_gitignore:
            matcher = GitIgnoreMatcher([p.pattern for p in current_gitignore.patterns])
            filtered_entries = []
            for entry in entries:
                relative_path = str(Path(entry.path).relative_to(root_path))
                relative_path = relative_path.replace('\\', '/')
                if matcher.should_include(relative_path):
                    filtered_entries.append(entry)
            entries = filtered_entries

        for i, entry in enumerate(entries):
            is_last = (i == len(entries) - 1)
            current_prefix = prefix_last if is_last else prefix
            output.append(''.join([*prefix_parts, current_prefix + entry.name]))

            if entry.is_dir():
                new_prefix_part = prefix_nothing if is_last else prefix_empty
                await _inner_generate(
                    Path(entry.path),
                    [*prefix_parts, new_prefix_part],
                    current_gitignore,
                    root_path,
                )

    root_gitignore = await _read_gitignore(path)
    output.append(os.path.basename(path))
    await _inner_generate(Path(path), [], root_gitignore, Path(path))
    return '\n'.join(output)


async def extract_html_from_webpage(url: str) -> str:
    """
    Extract clean text content from a webpage.

    Args:
        url: URL of the webpage to scrape

    Returns:
        Cleaned text content from the webpage

    Raises:
        aiohttp.ClientError: If there are network/HTTP errors
    """
    async with aiohttp.ClientSession() as session, session.get(url, timeout=5) as response:
        response.raise_for_status()
        return await response.text()


def clean_html_from_webpage(html: str) -> str:
    """
    Clean text extracted from a webpage.

    Args:
        html: Raw html from webpage

    Returns:
        Cleaned text with normalized spacing and removed boilerplate
    """
    # Noise patterns
    junk_elements = [
        'footer', 'header', 'nav', 'sidebar', 'ads', 'advertisement', 'promo', 'banner',
        'disclaimer', 'popup', 'cookie-banner', 'subscribe', 'share',
    ]
    junk_tags = [
        'footer', 'header', 'nav', 'aside', 'iframe', 'script', 'style', 'noscript', 'form',
        'button',
    ]
    # Function to clean unwanted noise
    def remove_junk(soup: BeautifulSoup) -> BeautifulSoup:
        # Remove unwanted tags
        for tag in junk_tags:
            for element in soup.find_all(tag):
                element.decompose()
        # Remove elements by class
        for junk_class in junk_elements:
            for element in soup.find_all(class_=junk_class):
                element.decompose()
        # Remove elements by id
        for junk_id in junk_elements:
            for element in soup.find_all(id=junk_id):
                element.decompose()
        return soup

    soup = BeautifulSoup(html, 'html.parser')
    soup = remove_junk(soup)
    text = soup.get_text(separator='\n', strip=True)
    # Step 1: Remove extra spaces
    text = re.sub(r' +', ' ', text)
    # Step 2: Replace newlines within paragraphs with spaces
    # Rule: If a newline is not followed by an uppercase letter or double newline,
    # replace it with a space, but preserve newlines after sentence-ending punctuation.
    text = re.sub(r'(?<!\n)\n(?!\n|[A-Z]|[.!?])', ' ', text)
    # Step 3: Normalize multiple newlines to single newlines for paragraphs
    text = re.sub(r'\n{2,}', '\n', text)
    # Step 4: Remove extra spaces and tabs but preserve newlines
    text = re.sub(r'[^\S\n]+', ' ', text)
    # Step 5: Remove trailing/leading whitespace from each line
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return '\n\n'.join(lines)


T = TypeVar('T')

class Registry:
    """
    A registry for managing different types of classes.
    Allows for registering classes with a type name and creating instances of these classes.
    The registry is case-insensitive for type names.
    """

    def __init__(self):
        """Initialize the registry with an empty dictionary."""
        self._registry: dict[str, type[T]] = {}

    @staticmethod
    def _clean_type_name(type_name: str | Enum) -> str:
        """Convert the type name to uppercase."""
        if isinstance(type_name, Enum):
            return type_name.name.upper()
        return type_name.upper()

    def register(self, type_name: str | Enum, item: type[T]) -> None:
        """
        Register a class with a specified type name (case-insensitive).

        If the type name is already registered, an assertion error is raised.

        Args:
            type_name:
                The type name to be associated with the class.

                The type_name is case-insensitive. If the type_name is an Enum, the name
                (`type_name.name`) is used.
            item: The class to be registered.
        """
        type_name = self._clean_type_name(type_name)
        assert type_name not in self._registry, f"Type '{type_name}' already registered."
        item._type_name = type_name
        self._registry[type_name] = item

    def get(self, type_name: str | Enum) -> type[T]:
        """
        Get the class associated with the given type name.

        Args:
            type_name: The type name of the class to retrieve.
        """
        type_name = self._clean_type_name(type_name)
        return self._registry[type_name]

    def __contains__(self, type_name: str | Enum) -> bool:
        """
        Check if a type name is registered in the registry (case insensitive).

        Args:
            type_name: The type name to check.
        """
        type_name = self._clean_type_name(type_name)
        return type_name in self._registry

    def create_instance(self, type_name: str | Enum, **data: dict) -> T:
        """
        Create an instance of the class associated with the given type name.

        Args:
            type_name (str): The type name of the class to instantiate.
            data: Keyword arguments to be passed to the class constructor.

        Raises:
            ValueError: If the type name is not registered in the registry.
        """
        if isinstance(type_name, Enum):
            type_name = type_name.name
        if type_name.upper() not in self._registry:
            raise ValueError(f"Unknown type `{type_name}`")
        return self._registry[type_name.upper()](**data)
