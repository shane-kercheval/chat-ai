"""Tests for the pdf module."""
import os
import time
import aiofiles
import aiohttp
import pytest
import tempfile
from pathlib import Path
from textwrap import dedent
from server.utilities import (
    clean_text_from_pdf,
    extract_jupyter_notebook_content,
    extract_text_from_pdf,
    generate_directory_tree,
    extract_html_from_webpage,
    clean_html_from_webpage,
)


# Test cases for positive matches (should truncate)
CLEAN_PDF_POSITIVE_TESTS = [
    ("This is a test.\nReferences", "This is a test."),
    ("This is a test.\nreferences", "This is a test."),
    ("This is a test.\nREFERENCES", "This is a test."),
    ("This is a test.\nBibliography", "This is a test."),
    ("This is a test.\nBIBLIOGRAPHY", "This is a test."),
    ("This is a test.\nWorks Cited", "This is a test."),
    ("This is a test.\nReferences:", "This is a test."),
    ("This is a test.\n1. References", "This is a test."),
    ("This is a test.\n2. Bibliography", "This is a test."),
    ("This is a test.\n   References   ", "This is a test."),
    ("This is a test.\n   1.   References   :   ", "This is a test."),
    ("This is a test.\nReferences\nRemove this text", "This is a test."),
    ("This is a test.\nreferences\nRemove this text", "This is a test."),
    ("This is a test.\nREFERENCES\nRemove this text", "This is a test."),
    ("This is a test.\nBibliography\nRemove this text", "This is a test."),
    ("This is a test.\nBIBLIOGRAPHY\nRemove this text", "This is a test."),
    ("This is a test.\nWorks Cited\nRemove this text", "This is a test."),
    ("This is a test.\nReferences:\nRemove this text", "This is a test."),
    ("This is a test.\n1. References\nRemove this text", "This is a test."),
    ("This is a test.\n2. Bibliography\nRemove this text", "This is a test."),
    ("This is a test.\n   References   \nRemove this text", "This is a test."),
    ("This is a test.\n   1.   References   :   \nRemove this text", "This is a test."),
]

# Test cases for negative matches (should not truncate)
CLEAN_PDF_NEGATIVE_TESTS = [
    ("This is a test.\nReferences and Further Reading", "This is a test.\n\nReferences and Further Reading"),  # noqa: E501
    ("This is a test.\nBibliography of Sources", "This is a test.\n\nBibliography of Sources"),
    ("This is a test.\nWorks Cited in This Paper", "This is a test.\n\nWorks Cited in This Paper"),
    ("This is a test.\n1. References and More", "This is a test.\n\n1. References and More"),
    ("This is a test.\nReferences: A Guide", "This is a test.\n\nReferences: A Guide"),
    ("References should not be removed", "References should not be removed"),
    ("This a test for references", "This a test for references"),
    ("See References.", "See References."),
]

@pytest.mark.asyncio
class TestExtractTextFromPdf:
    """Tests for extracting text from PDFs."""

    async def test__extract_text_from_pdf__from_url_and_file(self):
        ####
        # Test from URL
        ####
        attention_url = 'https://arxiv.org/pdf/1706.03762'
        text = await extract_text_from_pdf(attention_url)
        assert 'The dominant sequence transduction model' in text
        # save the text to a text file so we can track changes to how the text is extracted
        async with aiofiles.open('tests/test_files/pdf/attention_is_all_you_need__extracted_url.txt', 'w') as f:  # noqa: E501
            await f.write(text)

        ####
        # Test from file
        ####
        text_from_url = text
        attention_url = 'tests/test_files/pdf/attention_is_all_you_need_short.pdf'
        text = await extract_text_from_pdf(attention_url)
        assert text[0:100] == text_from_url[0:100]  # there will be small differences
        # save the text to a text file so we can track changes to how the text is extracted
        async with aiofiles.open('tests/test_files/pdf/attention_is_all_you_need__extracted_local.txt', 'w') as f:  # noqa: E501
            await f.write(text)

    async def test__clean_text_from_pdf(self):
        attention_url = 'https://arxiv.org/pdf/1706.03762'
        text = await extract_text_from_pdf(attention_url)
        assert 'The dominant sequence transduction model' in text
        cleaned_text = clean_text_from_pdf(text)

        assert cleaned_text.startswith('Provided proper attribution is provided')
        assert cleaned_text.endswith('fruitful comments, corrections and inspiration.')

        cleaned_text = clean_text_from_pdf(text, include_at='Abstract')
        assert cleaned_text.startswith('Abstract')

        cleaned_text = clean_text_from_pdf(text, exclude_at='Acknowledgements')
        assert cleaned_text.endswith("The code we used to train and evaluate our models is available at https://github.com/ tensorflow/tensor2tensor.")  # noqa

        cleaned_text = clean_text_from_pdf(
            text,
            include_at='Abstract',
            exclude_at='Acknowledgements',
        )
        assert cleaned_text.startswith('Abstract')
        assert cleaned_text.endswith("The code we used to train and evaluate our models is available at https://github.com/ tensorflow/tensor2tensor.")  # noqa
        # save the text to a text file so we can track changes to how the text is cleaned
        async with aiofiles.open('tests/test_files/pdf/attention_is_all_you_need_cleaned.txt', 'w') as f:  # noqa: E501
            await f.write(cleaned_text)

    async def test__clean_text_from_pdf__default(self):
        attention_url = 'https://arxiv.org/pdf/1706.03762'
        text = await extract_text_from_pdf(attention_url)
        assert 'The dominant sequence transduction model' in text
        cleaned_text = clean_text_from_pdf(text)
        async with aiofiles.open('tests/test_files/pdf/attention_is_all_you_need_cleaned__default.txt', 'w') as f:  # noqa: E501
            await f.write(cleaned_text)

    @pytest.mark.parametrize(("test_input", "expected_output"), CLEAN_PDF_POSITIVE_TESTS)
    async def test__clean_text_from_pdf__positive_truncation(self, test_input, expected_output):  # noqa: ANN001
        assert clean_text_from_pdf(test_input) == expected_output

    @pytest.mark.parametrize(("test_input", "expected_output"), CLEAN_PDF_NEGATIVE_TESTS)
    async def test__clean_text_from_pdf__negative_truncation(self, test_input, expected_output):  # noqa: ANN001
        assert clean_text_from_pdf(test_input) == expected_output


@pytest.mark.asyncio
class TestDirectoryTree:
    """Tests for directory tree generation."""

    async def test__generate_directory_tree__empty_directory(self):
        """Test generating tree for an empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tree = await generate_directory_tree(temp_dir)
            assert tree == os.path.basename(temp_dir)

    async def test__generate_directory_tree__single_level(self):
        """Test generating tree with files in a single directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "file1.txt").touch()
            Path(temp_dir, "file2.txt").touch()
            Path(temp_dir, ".hidden").touch()  # Should be ignored
            tree = await generate_directory_tree(temp_dir)
            expected = dedent(f"""
                {os.path.basename(temp_dir)}
                ├── file1.txt
                └── file2.txt""").strip()
            assert tree == expected

    async def test__generate_directory_tree__nested_directories(self):
        """Test generating tree with nested directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dir1 = Path(temp_dir, "dir1")
            dir2 = Path(temp_dir, "dir2")
            dir1.mkdir()
            dir2.mkdir()
            Path(dir1, "file1.txt").touch()
            Path(dir2, "file2.txt").touch()
            tree = await generate_directory_tree(temp_dir)
            expected = dedent(f"""
                {os.path.basename(temp_dir)}
                ├── dir1
                │   └── file1.txt
                └── dir2
                    └── file2.txt""").strip()
            assert tree == expected

    async def test__generate_directory_tree__deep_nesting(self):
        """Test generating tree with deeply nested directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            current = temp_dir
            for i in range(3):
                current = Path(current, f"level{i}")
                current.mkdir()
                Path(current, f"file{i}.txt").touch()
            tree = await generate_directory_tree(temp_dir)
            expected = dedent(f"""
                {os.path.basename(temp_dir)}
                └── level0
                    ├── file0.txt
                    └── level1
                        ├── file1.txt
                        └── level2
                            └── file2.txt""").strip()
            assert tree == expected

    async def test__generate_directory_tree__gitignore(self):
        """Test generating tree respects .gitignore."""
        with tempfile.TemporaryDirectory() as temp_dir:
            async with aiofiles.open(Path(temp_dir, ".gitignore"), "w") as f:
                await f.write("ignored.txt\n*.log")
            Path(temp_dir, "normal.txt").touch()
            Path(temp_dir, "ignored.txt").touch()  # Should be ignored
            Path(temp_dir, "test.log").touch()     # Should be ignored

            tree = await generate_directory_tree(temp_dir)
            expected = dedent(f"""
                {os.path.basename(temp_dir)}
                └── normal.txt""").strip()
            assert tree == expected

    async def test__generate_directory_tree__nested_gitignore(self):
        """Test handling of nested .gitignore files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Root .gitignore
            async with aiofiles.open(Path(temp_dir, ".gitignore"), "w") as f:
                await f.write("*.log")

            # Create nested structure
            subdir = Path(temp_dir, "subdir")
            subdir.mkdir()
            # Create test files
            Path(temp_dir, "root.txt").touch()     # Should show
            Path(temp_dir, "root.log").touch()     # Should be ignored
            Path(subdir, "sub.txt").touch()        # Should be ignored
            Path(subdir, "sub.log").touch()        # Should be ignored

            # should ignore .log but not .txt
            tree = await generate_directory_tree(temp_dir)
            expected = dedent(f"""
                {os.path.basename(temp_dir)}
                ├── root.txt
                └── subdir
                    └── sub.txt""").strip()
            assert tree == expected

            # now test the nested .gitignore
            # Subdir .gitignore
            async with aiofiles.open(Path(subdir, ".gitignore"), "w") as f:
                await f.write("*.txt")

            tree = await generate_directory_tree(temp_dir)
            expected = dedent(f"""
                {os.path.basename(temp_dir)}
                ├── root.txt
                └── subdir""").strip()
            assert tree == expected

    async def test__generate_directory_tree__special_characters(self):
        """Test handling of special characters in file/directory names."""
        with tempfile.TemporaryDirectory() as temp_dir:
            special_names = [
                "space name.txt",
                "unicode_★.txt",
                "symbols_!@#$%.txt",
                "quotes'\"quotes.txt",
                "brackets[].txt",
                "παράδειγμα.txt",  # Greek
                "例子.txt",        # Chinese
                "例え.txt",        # Japanese
            ]

            for name in special_names:
                Path(temp_dir, name).touch()

            tree = await generate_directory_tree(temp_dir)
            for name in special_names:
                assert name in tree

    async def test__generate_directory_tree__symlinks(self):
        """Test handling of symbolic links."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a target directory with files
            target_dir = Path(temp_dir, "target")
            target_dir.mkdir()
            Path(target_dir, "target_file.txt").touch()

            # Create a symlink to the directory
            link_path = Path(temp_dir, "link")
            os.symlink(target_dir, link_path)

            tree = await generate_directory_tree(temp_dir)
            expected = dedent(f"""
                {os.path.basename(temp_dir)}
                ├── link
                │   └── target_file.txt
                └── target
                    └── target_file.txt""").strip()
            assert tree == expected

    async def test__generate_directory_tree__complex_structure(self):
        """Test generating tree with a complex directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dir1 = Path(temp_dir, "dir1")
            dir2 = Path(temp_dir, "dir2")
            dir1.mkdir()
            dir2.mkdir()
            Path(temp_dir, "root1.txt").touch()
            Path(temp_dir, "root2.txt").touch()
            Path(dir1, "file1.txt").touch()
            subdir1 = Path(dir1, "subdir1")
            subdir1.mkdir()
            Path(subdir1, "sub1_file1.txt").touch()
            Path(dir2, "file2.txt").touch()
            Path(dir2, ".hidden.txt").touch()  # Should be ignored
            tree = await generate_directory_tree(temp_dir)
            expected = dedent(f"""
                {os.path.basename(temp_dir)}
                ├── dir1
                │   ├── file1.txt
                │   └── subdir1
                │       └── sub1_file1.txt
                ├── dir2
                │   └── file2.txt
                ├── root1.txt
                └── root2.txt""").strip()
            assert tree == expected

    async def test__generate_directory_tree__empty_directories(self):
        """Test generating tree with empty directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "empty1").mkdir()
            Path(temp_dir, "empty2").mkdir()
            Path(temp_dir, "dir1").mkdir()
            Path(temp_dir, "dir1", "empty3").mkdir()
            tree = await generate_directory_tree(temp_dir)
            expected = dedent(f"""
                {os.path.basename(temp_dir)}
                ├── dir1
                │   └── empty3
                ├── empty1
                └── empty2""").strip()
            assert tree == expected

    async def test__generate_directory_tree__large_directory(self):
        """Test performance with large directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # add gitignore to root for log files
            async with aiofiles.open(Path(temp_dir, ".gitignore"), "w") as f:
                await f.write("*.log")
            # Create many files and directories
            for i in range(100):
                subdir = Path(temp_dir, f"dir_{i}")
                subdir.mkdir()
                for j in range(10):
                    Path(subdir, f"file_{j}.txt").touch()
                    Path(subdir, f"file_{j}.ignore").touch()
                    Path(subdir, f"file_{j}.log").touch()
                    # add gitignore to every directory
                    async with aiofiles.open(Path(subdir, ".gitignore"), "w") as f:
                        await f.write("*.ignore")

            start_time = time.time()
            tree = await generate_directory_tree(temp_dir)
            duration = time.time() - start_time
            # Verify it completes in reasonable time
            assert duration < 0.5
            assert tree.count("dir_") == 100
            assert tree.count("file_") == 1000

    async def test__generate_directory_tree__non_existent_directory(self):
        """Test generating tree for non-existent directory raises error."""
        with pytest.raises(ValueError, match="Path is not a directory"):
            await generate_directory_tree("/non/existent/path")

    async def test__generate_directory_tree__file_path(self):
        """Test generating tree for a file path raises error."""
        with tempfile.NamedTemporaryFile() as temp_file:  # noqa: SIM117
            with pytest.raises(ValueError, match="Path is not a directory"):
                await generate_directory_tree(temp_file.name)

    async def  test__generate_directory_tree__current_directory(self):
        """Test getting a simple directory resource."""
        # get project directory
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tree = await generate_directory_tree(project_dir)
        print(tree[0:2000])
        assert len(tree) < 10_000
        assert 'node_modules' not in tree

    async def test__node_modules_issue(self):
        """Test the specific case of node_modules being properly excluded."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            # Create test directory structure
            (tmp_path / 'client').mkdir()
            (tmp_path / 'client/node_modules').mkdir()
            (tmp_path / 'client/node_modules/@electron').mkdir(parents=True)
            (tmp_path / 'client/node_modules/@electron/file.txt').touch()
            (tmp_path / 'client/src').mkdir()
            (tmp_path / 'client/src/main.ts').touch()

            # Create .gitignore
            async with aiofiles.open(tmp_path / '.gitignore', 'w') as f:
                await f.write('client/node_modules\n')

            tree = await generate_directory_tree(str(tmp_path))

        assert 'node_modules' not in tree
        assert '@electron' not in tree
        assert os.path.basename(temp_dir) in tree
        assert 'src' in tree
        assert 'main.ts' in tree

    @pytest.mark.parametrize(
        ('gitignore', 'should_include', 'should_exclude'),
        [
        # Exact directory match (relative and absolute paths)
        ('dir1', ['dir2', 'dir3', 'subdir', 'file2.txt', 'file3.txt'], ['dir1', 'file1.txt']),
        ('/dir1', ['dir2', 'dir3', 'subdir', 'file2.txt', 'file3.txt'], ['dir1', 'file1.txt']),

        # Trailing slash (contents excluded, directory remains)
        ('dir2/', ['dir1', 'dir2', 'dir3', 'subdir', 'file1.txt', 'file3.txt'], ['file2.txt']),

        # Nested paths
        ('dir3/subdir', ['dir1', 'dir2', 'dir3', 'file1.txt', 'file2.txt'], ['subdir', 'file3.txt']),  # noqa: E501
        ('dir3/subdir/', ['dir1', 'dir2', 'dir3', 'subdir', 'file1.txt', 'file2.txt'], ['file3.txt']),  # noqa: E501

        # Multiple patterns
        ('dir1\ndir2', ['dir3', 'subdir', 'file3.txt'], ['dir1', 'dir2', 'file1.txt', 'file2.txt']),  # noqa: E501
        ('dir1/\ndir3/subdir/', ['dir1', 'dir2', 'dir3', 'subdir', 'file2.txt'], ['file1.txt', 'file3.txt']),  # noqa: E501

        # Wildcard matches everything
        ('*', [], ['dir1', 'dir2', 'dir3', 'subdir', 'file1.txt', 'file2.txt', 'file3.txt']),

        # Negation
        ('*\n!dir1', ['dir1', 'file1.txt'], ['dir2', 'dir3', 'file2.txt', 'subdir', 'file3.txt']),
        ('*\n!dir3/subdir', ['dir3', 'subdir', 'file3.txt'], ['dir1', 'dir2', 'file1.txt', 'file2.txt']),  # noqa: E501

        # Comments
        ('#dir1\n#dir2', ['dir1', 'dir2', 'dir3', 'subdir', 'file1.txt', 'file2.txt', 'file3.txt'], []),  # noqa: E501
        ('#dir1\ndir2', ['dir1', 'dir3', 'subdir', 'file1.txt', 'file3.txt'], ['dir2', 'file2.txt']),  # noqa: E501

        # Empty pattern (should not ignore anything)
        ('', ['dir1', 'dir2', 'dir3', 'file1.txt', 'file2.txt', 'subdir', 'file3.txt'], []),

        # Edge case: Slash at start (should match root only)
        ('/dir1/file1.txt', ['dir1', 'dir2', 'dir3', 'file2.txt', 'subdir', 'file3.txt'], ['file1.txt']),  # noqa: E501

        # Edge case: Exclude files with wildcards
        ('*.txt', ['dir1', 'dir2', 'dir3', 'subdir'], ['file1.txt', 'file2.txt', 'file3.txt']),

        # Edge case: Include specific file after wildcard exclusion
        ('*.txt\n!dir1/file1.txt', ['dir1', 'dir2', 'dir3', 'subdir', 'file1.txt'], ['file2.txt', 'file3.txt']),  # noqa: E501
    ],
    )
    async def test_generate_directory_tree__gitignore_patterns(self, gitignore, should_include, should_exclude):  # noqa: ANN001, E501
        """Test various gitignore pattern formats."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            # Create test directory structure
            (tmp_path / "dir1").mkdir()
            (tmp_path / "dir1/file1.txt").touch()
            (tmp_path / "dir2").mkdir()
            (tmp_path / "dir2/file2.txt").touch()
            (tmp_path / "dir3").mkdir()
            (tmp_path / "dir3/subdir").mkdir()
            (tmp_path / "dir3/subdir/file3.txt").touch()
            # Write the .gitignore file
            async with aiofiles.open(tmp_path / ".gitignore", "w") as f:
                await f.write(gitignore)
            # Generate the directory tree
            tree = await generate_directory_tree(str(tmp_path))

        # print(f"Pattern: `\n{gitignore}\n`")
        # print(f"Expected inclusions: `{should_include}`")
        # print(f"Expected exclusions: `{should_exclude}`")
        # print(f"Actual tree: `\n{tree}\n`")
        # Check inclusions
        for include in should_include:
            assert include in tree, f"Pattern '{gitignore}' should not exclude '{include}'"
        # Check exclusions
        for exclude in should_exclude:
            assert exclude not in tree, f"Pattern '{gitignore}' should exclude '{exclude}'"

    async def test__generate_directory_tree__nested_gitignore__with_negate(self):
        """Test handling of nested .gitignore files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            # Create test structure
            (tmp_path / 'parent').mkdir()
            (tmp_path / 'parent/child').mkdir()
            (tmp_path / 'parent/child/grandchild').mkdir()
            (tmp_path / 'parent/file1.txt').touch()
            (tmp_path / 'parent/child/file2.txt').touch()
            (tmp_path / 'parent/child/grandchild/file3.txt').touch()
            # Root .gitignore ignores all .txt files
            async with aiofiles.open(tmp_path / '.gitignore', 'w') as f:
                await f.write('*.txt\n')
            # Child .gitignore negates the pattern for its directory
            async with aiofiles.open(tmp_path / 'parent/child/.gitignore', 'w') as f:
                await f.write('!*.txt\n')
            tree = await generate_directory_tree(str(tmp_path))

        assert 'parent' in tree
        # excluded from root .gitignore
        assert 'file1.txt' not in tree
        # Child .txt should be included
        assert 'file2.txt' in tree
        assert 'file3.txt' in tree

    async def test__complex_directory_structure(self):
        """Test a complex directory structure with multiple gitignore patterns."""
        # Create a complex directory structure similar to a real project
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            (tmp_path / 'src').mkdir()
            (tmp_path / 'src/main.ts').touch()
            (tmp_path / 'dist').mkdir()
            (tmp_path / 'dist/main.js').touch()
            (tmp_path / 'node_modules').mkdir()
            (tmp_path / 'node_modules/package').mkdir()
            (tmp_path / 'node_modules/package/index.js').touch()
            (tmp_path / 'tests').mkdir()
            (tmp_path / 'tests/test.ts').touch()
            (tmp_path / 'coverage').mkdir()
            (tmp_path / 'coverage/report.html').touch()
            # Create .gitignore with typical patterns
            async with aiofiles.open(tmp_path / '.gitignore', 'w') as f:
                await f.write(dedent("""
                    node_modules
                    dist
                    coverage
                    *.log
                """).strip())
            tree = await generate_directory_tree(str(tmp_path))
        # Check that ignored directories are excluded
        assert 'node_modules' not in tree
        assert 'dist' not in tree
        assert 'coverage' not in tree
        assert 'main.js' not in tree
        assert 'package' not in tree
        assert 'index.js' not in tree
        assert 'report.html' not in tree
        # Check that non-ignored directories are included
        assert 'src' in tree
        assert 'main.ts' in tree
        assert 'tests' in tree
        assert 'test.ts' in tree


@pytest.mark.asyncio
class TestExtractHtmlFromUrl:
    """Tests for extracting text from Wepage."""

    async def test__extract_html_from_webpage__simple_html(self):
        """Test extracting text from basic HTML."""
        text = await extract_html_from_webpage("https://example.com")
        # Example.com has consistent content
        assert '<body>' in text
        assert "Example Domain" in text
        assert "This domain is for use in illustrative examples" in text

    async def test__extract_html_from_webpage__invalid_url(self):
        """Test extracting text from invalid URL."""
        with pytest.raises(aiohttp.client_exceptions.InvalidUrlClientError):
            await extract_html_from_webpage("not_a_url")


class TestWebpageTextCleaning:
    """Tests for web page text cleaning."""

    def test__clean_html_from_webpage__removes_junk_tags(self):
        """Test removal of unwanted HTML tags."""
        html = """
        <html>
            <body>
                <h1>Main Content</h1>
                <p>Important text.</p>
                <script>alert('test');</script>
                <style>.test { color: red; }</style>
                <footer>Copyright 2024</footer>
                <nav>Menu items</nav>
                <form>Subscribe form</form>
            </body>
            <footer>Copyright 2024</footer>
            <nav>Menu items</nav>
        </html>
        """
        text = clean_html_from_webpage(html)
        assert "Main Content" in text
        assert "Important text" in text
        assert "alert('test')" not in text
        assert "color: red" not in text
        assert "Copyright" not in text
        assert "Menu items" not in text
        assert "Subscribe form" not in text

    def test__clean_html_from_webpage__removes_junk_classes(self):
        """Test removal of elements with unwanted classes."""
        html = """
        <html>
            <body>
                <h1>Main Content</h1>
                <div class="content">Keep this</div>
                <div class="footer">Remove this</div>
                <div class="sidebar">Remove this</div>
                <div class="ads">Remove this</div>
                <div class="promo">Remove this</div>
            </body>
        </html>
        """
        text = clean_html_from_webpage(html)
        assert "Main Content" in text
        assert "Keep this" in text
        assert "Remove this" not in text

    def test__clean_html_from_webpage__removes_junk_ids(self):
        """Test removal of elements with unwanted IDs."""
        html = """
        <html>
            <body>
                <h1>Main Content</h1>
                <div id="content">Keep this</div>
                <div id="footer">Remove this</div>
                <div id="sidebar">Remove this</div>
                <div id="ads">Remove this</div>
            </body>
        </html>
        """
        text = clean_html_from_webpage(html)
        assert "Main Content" in text
        assert "Keep this" in text
        assert "Remove this" not in text

    def test__clean_html_from_webpage__preserves_paragraph_structure(self):
        """Test that paragraph structure is preserved."""
        html = """
        <html>
            <body>
                <h1>Title   </h1>
                <p>First    paragraph with multiple sentences.
                This is still the first paragraph.
                This is another. This
                is another.</p>
                <p> Second paragraph.</p>
            </body>
        </html>
        """
        text = clean_html_from_webpage(html)
        expected = (
            "Title\n\n"
            "First paragraph with multiple sentences. This is still the first paragraph. This is another. This is another.\n\n"  # noqa: E501
            "Second paragraph."
        )
        assert text == expected

    def test__clean_html_from_webpage__sentence_continuations(self):
        """Test handling of sentence continuations and line breaks."""
        html = """
        <html>
            <body>
                <!-- Basic sentence continuation -->
                <p>This is a sentence
                that continues here.</p>

                <!-- Multiple sentence continuation -->
                <p>This is a longer sentence
                that continues here
                and here too.</p>

                <!-- Proper sentence breaks -->
                <p>This is one sentence.
                This is another sentence.</p>

                <!-- Mixed cases -->
                <p>First sentence.
                Second sentence.
                this is a continuation
                of the second sentence.

                But this is a new paragraph even within the same tag.
                </p>

                <!-- Parenthetical continuation -->
                <p>This is a sentence
                (with a parenthetical).</p>

                <!-- Bracket continuation -->
                <p>This is a sentence
                [with a bracket].</p>

                <!-- Multiple punctuation cases -->
                <p>Question?
                New sentence.
                this continues the sentence.</p>

                <p>Exclamation!
                New sentence.
                this continues the sentence.</p>
            </body>
        </html>
        """
        text = clean_html_from_webpage(html)
        expected = (
            "This is a sentence that continues here.\n\n"
            "This is a longer sentence that continues here and here too.\n\n"
            "This is one sentence. This is another sentence.\n\n"
            "First sentence. Second sentence. this is a continuation of the second sentence.\n\n"
            "But this is a new paragraph even within the same tag.\n\n"
            "This is a sentence (with a parenthetical).\n\n"
            "This is a sentence [with a bracket].\n\n"
            "Question? New sentence. this continues the sentence.\n\n"
            "Exclamation! New sentence. this continues the sentence."
        )
        assert text == expected

    def test__clean_html_from_webpage__handles_special_characters(self):
        """Test handling of special characters and HTML entities."""
        html = """
        <html>
            <body>
                <p>Special chars: &amp; &lt; &gt; &quot; &copy; &euro;</p>
                <p>Unicode: 你好 καλημέρα</p>
            </body>
        </html>
        """
        text = clean_html_from_webpage(html)
        assert "Special chars: & < > \" © €" in text
        assert "Unicode: 你好 καλημέρα" in text

    def test__clean_html_from_webpage__normalizes_whitespace(self):
        """Test whitespace normalization."""
        html = """
        <html>
            <body>
                <p>Multiple    spaces</p>
                <p>Multiple

                paragraphs</p>
                <p>Trailing space </p>
                <p> Leading space</p>
            </body>
        </html>
        """
        text = clean_html_from_webpage(html)
        expected = (
            "Multiple spaces\n\n"
            "Multiple\n\n"
            "paragraphs\n\n"
            "Trailing space\n\n"
            "Leading space"
        )
        assert text == expected

    def test__clean_html_from_webpage__handles_nested_elements(self):
        """Test cleaning of nested elements."""
        html = """
        <html>
            <body>
                <div class="content">
                    <h1>Keep this</h1>
                    <div class="footer">
                        <p>Remove this</p>
                        <div>And this</div>
                    </div>
                    <p>Keep this too</p>
                </div>
            </body>
        </html>
        """
        text = clean_html_from_webpage(html)
        assert "Keep this" in text
        assert "Keep this too" in text
        assert "Remove this" not in text
        assert "And this" not in text

    def test__clean_html_from_webpage__handles_empty_input(self):
        """Test handling of empty or whitespace-only input."""
        assert clean_html_from_webpage("") == ""
        assert clean_html_from_webpage("   ") == ""
        assert clean_html_from_webpage("<html></html>") == ""
        assert clean_html_from_webpage("<html><body>  </body></html>") == ""

    def test__clean_html_from_webpage__preserves_sentence_structure(self):
        """Test that sentence structure is preserved appropriately."""
        html = """
        <html>
            <body>
                <p>First sentence. Second sentence!</p>
                <p>Question? Answer.</p>
                <p>Reference [1]. Next sentence.</p>
                <p>lower case
                continuation of sentence.</p>
            </body>
        </html>
        """
        text = clean_html_from_webpage(html)
        assert "First sentence. Second sentence!" in text
        assert "Question? Answer." in text
        assert "Reference [1]. Next sentence." in text
        assert "lower case continuation of sentence." in text


class TestJupyterNotebookContent:
    """Tests for extracting content from Jupyter notebooks."""

    def test__extract_jupyter_notebook_content__simple(self):
        """Test extracting content from a simple Jupyter notebook."""
        test_file = "tests/test_files/notebooks/simple_notebook.ipynb"
        content = extract_jupyter_notebook_content(test_file)
        print(content)
        # test headers
        assert '[MARKDOWN CELL]' in content
        assert '[CODE CELL]' in content
        assert '[CODE CELL OUTPUT]' in content
        # test markdown cell
        assert "This is a fake notebook." in content
        assert "The goal is to test extraction." in content

        # test code cell and output
        assert 'print(f"Hello {123}")' in content
        assert "Hello 123" in content

        # test error handling
        assert "import asdf" in content
        assert "ModuleNotFoundError" in content

        # ensure that the empty cells are not included
        assert "CODE CELL]\n\n\n" not in content
        assert "MARKDOWN CELL]\n\n\n" not in content

        # test matplotlib output
        assert "matplotlib.pyplot" in content
        assert "<Figure size" in content

        # test pandas dataframe output
        assert "pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})" in content
        assert "a  b\n0  1  4\n1  2  5\n2  3  6" in content
