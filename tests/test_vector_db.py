"""Tests for VectorDB functionality."""
import pytest
import numpy as np
from textwrap import dedent
from sentence_transformers import SentenceTransformer
from server.vector_db import SimilarityScorer, Document, Chunk, chunk_text


class MockSentenceTransformer:
    """Mock model that returns predictable embeddings."""

    def __init__(self):
        self.encode_count = 0

    def encode(self, texts: str | list[str]) -> np.ndarray | list[np.ndarray]:
        """
        Return embeddings that give predictable similarity scores.

        Single texts (queries) return [1, 0, 0]
        Lists of texts (chunks) return embeddings that will give known cosine similarities:
        - 'test': [0.5, 0.866, 0] -> 0.5 similarity
        - 'similar': [0.866, 0.5, 0] -> 0.866 similarity
        - 'same': [1, 0, 0] -> 1.0 similarity
        - Other: [0, 1, 0] -> 0.0 similarity
        """
        self.encode_count += 1
        if isinstance(texts, str):
            return np.array([1.0, 0.0, 0.0])
        embeddings = []
        for text in texts:
            if 'test' in text.lower():
                vec = [0.5, 0.866, 0]  # 0.5 similarity
            elif 'similar' in text.lower():
                vec = [0.866, 0.5, 0]  # 0.866 similarity
            elif 'same' in text.lower():
                vec = [1, 0, 0]        # 1.0 similarity
            else:
                vec = [0, 1, 0]        # 0.0 similarity
            embeddings.append(np.array(vec))
        return np.array(embeddings)


@pytest.fixture
def sample_documents():
    """Create sample documents with known similarity patterns."""
    return [
        Document(
            key="doc1",
            content=dedent("""
                This is a test document. It has some test content.
                This should give 0.5 score.
                
                This is similar content that should score higher.
                This is the same as the query for max score.
                
                This is other content with no match.
            """).strip(),  # noqa: W293
            last_modified="2024-01-01",
        ),
        Document(
            key="doc2",
            content=dedent("""
                Another document with test content.
                More similar content here.
                
                Some unrelated text that won't match.
            """).strip(),  # noqa: W293
            last_modified="2024-01-01",
        ),
    ]


class TestSimilarityScorer:
    """Test SimilarityScorer functionality."""

    def test__initialization(self):
        """Test initializing scorer."""
        scorer = SimilarityScorer(MockSentenceTransformer(), cache_size=5)
        assert scorer.chunk_size == 500  # default
        scorer = SimilarityScorer(MockSentenceTransformer(), cache_size=10, chunk_size=200)
        assert scorer.chunk_size == 200

    @pytest.mark.parametrize('chunk_size', [5, 20, 50, 100, 500])
    def test__get_document_chunks_single_doc(self, sample_documents, chunk_size):  # noqa: ANN001
        """Test getting chunks from a single document."""
        scorer = SimilarityScorer(model=MockSentenceTransformer(), cache_size=5, chunk_size=chunk_size)  # noqa: E501
        chunks = scorer._get_document_chunks([sample_documents[0]])
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.key == "doc1"
            assert isinstance(chunk.embedding, np.ndarray)
            assert chunk.embedding.shape == (3,)

    @pytest.mark.parametrize('chunk_size', [5, 20, 50, 100, 500])
    def test__caching(self, sample_documents, chunk_size):  # noqa: ANN001
        """Test chunk caching."""
        # First call should compute embeddings
        scorer = SimilarityScorer(MockSentenceTransformer(), chunk_size=chunk_size)
        chunks1 = scorer._get_document_chunks([sample_documents[0]])
        initial_encode_count = scorer.model.encode_count
        # Second call should use cache
        chunks2 = scorer._get_document_chunks([sample_documents[0]])
        assert scorer.model.encode_count == initial_encode_count
        # Verify chunks are identical
        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2):
            assert c1.key == c2.key
            assert c1.text == c2.text
            assert np.array_equal(c1.embedding, c2.embedding)

    @pytest.mark.parametrize('chunk_size', [5, 20, 50, 100, 500])
    def test__cache_invalidation(self, sample_documents, chunk_size):  # noqa: ANN001
        """Test cache invalidation on document change."""
        doc = sample_documents[0]
        # Initial chunking
        scorer = SimilarityScorer(MockSentenceTransformer(), chunk_size=chunk_size)
        chunks1 = scorer._get_document_chunks([doc])
        initial_count = scorer.model.encode_count
        # Same document, different modification time
        modified_doc = Document(doc.key, doc.content, "2024-01-02")
        chunks2 = scorer._get_document_chunks([modified_doc])
        assert scorer.model.encode_count > initial_count
        assert [c.text for c in chunks1] == [c.text for c in chunks2]

    def test_complete_caching_behavior(self):
        """Test comprehensive caching behavior across multiple documents."""
        # Create documents with known number of chunks
        docs = [
            Document(
                "doc1.txt",
                ". ".join(["Test sentence one"] * 5),  # Will create multiple chunks
                "2024-01-01",
            ),
            Document(
                "doc2.txt",
                ". ".join(["Test sentence two"] * 5),
                "2024-01-01",
            ),
        ]

        # First request - should cache all chunks
        scorer = SimilarityScorer(MockSentenceTransformer(), chunk_size=20, cache_size=5)
        initial_chunks = scorer._get_document_chunks(docs)
        initial_chunk_count = len(initial_chunks)
        initial_encode_count = scorer.model.encode_count

        # Map of chunks by document
        initial_chunks_by_doc = {}
        for chunk in initial_chunks:
            if chunk.key not in initial_chunks_by_doc:
                initial_chunks_by_doc[chunk.key] = []
            initial_chunks_by_doc[chunk.key].append(chunk)

        # Verify each document's chunks are cached
        for doc_key, doc_chunks in initial_chunks_by_doc.items():
            cached_chunks = scorer._cache[f"{doc_key}_2024-01-01"]
            assert len(cached_chunks) == len(doc_chunks)
            for c1, c2 in zip(cached_chunks, doc_chunks):
                assert c1.text == c2.text
                assert np.array_equal(c1.embedding, c2.embedding)

        assert ' '.join([c.text for c in initial_chunks_by_doc['doc1.txt']]) == docs[0].content
        assert ' '.join([c.text for c in initial_chunks_by_doc['doc2.txt']]) == docs[1].content

        # Second request - should use cache completely
        second_chunks = scorer._get_document_chunks(docs)
        assert scorer.model.encode_count == initial_encode_count  # No new encodings
        assert len(second_chunks) == initial_chunk_count

        # Verify all chunks are identical
        for c1, c2 in zip(sorted(initial_chunks, key=lambda x: x.text), sorted(second_chunks, key=lambda x: x.text), strict=True):  # noqa: E501
            assert c1.text == c2.text
            assert np.array_equal(c1.embedding, c2.embedding)

    def test_partial_cache_invalidation(self):
        """Test cache invalidation when some documents change."""
        docs = [
            Document("doc1.txt", "Test content one", "2024-01-01"),
            Document("doc2.txt", "Test content two", "2024-01-01"),
        ]
        scorer = SimilarityScorer(MockSentenceTransformer(), chunk_size=20)
        _ = scorer._get_document_chunks(docs)
        initial_encode_count = scorer.model.encode_count
        # Modify one document
        modified_docs = [
            Document("doc1.txt", "Test content one", "2024-01-02"),  # Changed timestamp
            Document("doc2.txt", "Test content two", "2024-01-01"),  # Same
        ]
        # Get chunks again
        _ = scorer._get_document_chunks(modified_docs)
        # Verify:
        # 1. Only modified document caused new encodings
        assert scorer.model.encode_count == initial_encode_count + 1  # One new encode call
        # 2. Cache still contains unmodified document
        assert "doc2.txt_2024-01-01" in scorer._cache
        # 3. Cache has updated modified document
        assert "doc1.txt_2024-01-02" in scorer._cache

    def test__similarity_matrix_calculation(self):
        """Test cosine similarity matrix calculation."""
        scorer = SimilarityScorer(MockSentenceTransformer())
        query_embedding = np.array([1, 0, 0])
        chunk_embeddings = np.array([
            [1, 0, 0],    # Should give 1.0
            [0, 1, 0],    # Should give 0.0
            [0.5, 0.866, 0],  # Should give 0.5
        ])
        similarities = scorer._cosine_similarity_matrix(query_embedding, chunk_embeddings)
        assert len(similarities) == 3
        assert np.isclose(similarities[0], 1.0)
        assert np.isclose(similarities[1], 0.0)
        assert np.isclose(similarities[2].round(4), 0.5)

    def test_similarity_edge_cases(self):
        """Test similarity calculations with edge case vectors."""
        # Test zero vectors
        zero_vec = np.zeros(3)
        normal_vec = np.array([1, 0, 0])
        chunk_embeddings = np.vstack([zero_vec, normal_vec])
        scorer = SimilarityScorer(MockSentenceTransformer())
        similarities = scorer._cosine_similarity_matrix(normal_vec, chunk_embeddings)
        assert not np.isnan(similarities).any()  # No NaN values
        assert np.isclose(similarities[0], 0.0)
        assert np.isclose(similarities[1], 1.0)  # Normal case works

    @pytest.mark.parametrize('chunk_size', [5, 20])
    def test__score_results(self, sample_documents, chunk_size):  # noqa: ANN001
        """Test scoring results are as expected."""
        scorer = SimilarityScorer(model=MockSentenceTransformer(), chunk_size=chunk_size)
        results = scorer.score(sample_documents, "query")
        # Verify results structure
        assert len(results) > 0
        for chunk in results:
            assert isinstance(chunk.score, float)
            assert 0 <= chunk.score <= 1
            if 'same' in chunk.text.lower():
                assert np.isclose(chunk.score, 1.0)
            elif 'similar' in chunk.text.lower():
                assert np.isclose(round(chunk.score, 4), 0.866)
            elif 'test' in chunk.text.lower():
                assert np.isclose(round(chunk.score, 4), 0.5)
            else:
                assert np.isclose(chunk.score, 0.0)

    def test__empty_documents(self):
        """Test handling empty documents."""
        scorer = SimilarityScorer(MockSentenceTransformer())
        results = scorer.score([Document("empty.txt", "", "2024-01-01")], "query")
        assert len(results) == 0

    @pytest.mark.parametrize('chunk_size', [5, 20, 50, 100])
    def test__multiple_documents(self, sample_documents, chunk_size):  # noqa: ANN001
        """Test scoring across multiple documents."""
        scorer = SimilarityScorer(model=MockSentenceTransformer(), chunk_size=chunk_size)
        results = scorer.score(sample_documents, "query")
        for chunk in results:
            assert isinstance(chunk.score, float)
            assert 0 <= chunk.score <= 1
        # Verify we got results from both documents
        doc1_chunks = [c for c in results if c.key == "doc1"]
        doc2_chunks = [c for c in results if c.key == "doc2"]
        assert len(doc1_chunks) > 0
        assert len(doc2_chunks) > 0
        assert ' '.join(c.text for c in doc1_chunks) == ' '.join(sample_documents[0].content.split())  # noqa: E501
        assert ' '.join(c.text for c in doc2_chunks) == ' '.join(sample_documents[1].content.split())  # noqa: E501

    def test_cache_eviction(self):
        """Test that cache evicts items when size limit is reached."""
        small_cache_scorer = SimilarityScorer(MockSentenceTransformer(), cache_size=2)
        # Create more documents than cache size
        docs = [
            Document(f"doc{i}.txt", f"Test content {i}", f"2024-01-0{i}")
            for i in range(4)
        ]
        # Process all documents
        for doc in docs:
            small_cache_scorer._get_document_chunks([doc])
        # Verify only most recent are cached
        assert len(small_cache_scorer._cache) == 2
        assert "doc2.txt_2024-01-02" in small_cache_scorer._cache
        assert "doc3.txt_2024-01-03" in small_cache_scorer._cache


class TestWithRealTransformer:
    """Test with actual SentenceTransformer."""

    @pytest.fixture
    def real_scorer(self):
        return SimilarityScorer(SentenceTransformer('all-MiniLM-L6-v2'), chunk_size=500)

    def test__real_embeddings(self, real_scorer):  # noqa: ANN001
        """Test embedding generation with real model."""
        doc = Document(
            "test.txt",
            "This is a test document about machine learning.",
            "2024-01-01",
        )
        chunks = real_scorer._get_document_chunks([doc])
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.embedding.shape == (384,)  # real model dimension

    def test__real_scoring(self, real_scorer):  # noqa: ANN001
        """Test scoring with real embeddings."""
        docs = [
            Document(
                "ml.txt",
                "Machine learning is a branch of artificial intelligence.",
                "2024-01-01",
            ),
            Document(
                "db.txt",
                "Databases store and organize data efficiently.",
                "2024-01-01",
            ),
        ]
        # ML-related query should score higher with ML content
        results = real_scorer.score(docs, "artificial intelligence")
        ml_scores = [c.score for c in results if "machine learning" in c.text.lower()]
        db_scores = [c.score for c in results if "database" in c.text.lower()]
        assert max(ml_scores) > max(db_scores)


class TestChunkText:
    """Tests for text chunking functionality."""

    def test___chunk_text__empty_input(self):
        """Test chunking with empty input."""
        assert chunk_text("", key="test") == []
        assert chunk_text(None, key="test") == []

    @pytest.mark.parametrize('chunk_size', [5, 100, len("This is a short sentence.")])
    def test___chunk_text__single_sentence_single_chunk(self, chunk_size: int):
        """Test chunking with a single sentence above/below/equal to chunk size."""
        text = "This is a short sentence."
        chunks = chunk_text(text, key="test", chunk_size=chunk_size)
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].index == 0
        assert chunks[0].key == "test"

    def test___chunk_text__multiple_sentences_within_chunk(self):
        """Test multiple sentences that fit within one chunk."""
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunk_text(text, key="test", chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0].text == text

    def test___chunk_text__basic_chunking(self):
        """Test basic chunking with multiple chunks."""
        text = dedent("""
            First sentence in chunk one. Second sentence in chunk one.
            First sentence in chunk two. Second sentence in chunk two.
            First sentence in chunk three. Second sentence in chunk three.
        """).strip()
        chunks = chunk_text(text, key="test", chunk_size=50)
        assert len(chunks) > 1
        # Verify each chunk is roughly the target size
        for chunk in chunks:
            assert len(chunk.text) <= 50 or chunk.text.count('.') == 1

    @pytest.mark.parametrize('chunk_size', [5, 20, 50, 100])
    def test___chunk_text__sentence_boundaries(self, chunk_size: int):
        """Test that chunks respect sentence boundaries."""
        text = dedent("""
            First sentence. Second sentence.
            Third sentence. Fourth sentence.
            Fifth sentence. Sixth sentence.
        """).strip()

        chunks = chunk_text(text, key="test", chunk_size=chunk_size)
        # Verify each chunk ends with a complete sentence
        for chunk in chunks:
            assert chunk.text.strip().endswith('.')

    def test___chunk_text__large_sentences(self):
        """Test handling of sentences larger than chunk size."""
        text = "This is a very very very very very very very very very very long sentence. This is another sentence."  # noqa: E501
        chunks = chunk_text(text, key="test", chunk_size=20)
        # Should keep large sentences together
        assert len(chunks) == 2
        assert chunks[0].text.endswith('sentence.')

    @pytest.mark.parametrize('chunk_size', [5, 20, 50, 100])
    def test___chunk_text__special_punctuation(self, chunk_size: int):
        """Test chunking with various punctuation marks."""
        text = dedent("""
            First sentence! Second sentence?
            Third sentence... Fourth sentence:
            Fifth sentence; Sixth sentence.
        """).strip()

        chunks = chunk_text(text, key="test", chunk_size=chunk_size)
        all_text = ' '.join(c.text for c in chunks)
        assert '!' in all_text
        assert '?' in all_text
        assert '...' in all_text

    @pytest.mark.parametrize('chunk_size', [5, 20, 50])
    def test___chunk_text__whitespace_handling(self, chunk_size: int):
        """Test proper handling of various whitespace."""
        text = dedent("""
            First    sentence.    
            
            Second         sentence.
            
            Third    sentence.
        """).strip()  # noqa: W291, W293

        chunks = chunk_text(text, key="test", chunk_size=chunk_size)
        # Verify excess whitespace is normalized
        for chunk in chunks:
            assert "    " not in chunk.text  # No multiple spaces
            assert "\n\n" not in chunk.text  # No multiple newlines
            assert "\t" not in chunk.text    # No tabs

        # Verify content is preserved after normalization
        all_text = ' '.join(c.text for c in chunks)
        assert "First sentence." in all_text
        assert "Second sentence." in all_text
        assert "Third sentence." in all_text

    def test___chunk_text__indexes_are_sequential(self):
        """Test that chunk indexes are sequential."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunk_text(text, key="test", chunk_size=30)
        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    @pytest.mark.parametrize('chunk_size', [5, 20, 50, 100, 1000])
    def test___chunk_text__preserves_content(self, chunk_size: int):
        """Test that no content is lost in chunking process."""
        text = dedent("""
            ChromaDB is a document database that makes it easy to store and query
            embeddings for natural language processing tasks. It provides a simple API
            for storing documents and their embeddings, and for querying those embeddings
            to find similar documents. ChromaDB is built on top of SQLite and supports
            multiple embedding models, including OpenAI's embeddings and Sentence Transformers.

            One of the key features of ChromaDB is its ability to efficiently store and
            query high-dimensional vectors. It uses approximate nearest neighbor search
            algorithms to quickly find similar documents without having to compare every
            document in the database.
        """).strip()

        chunks = chunk_text(text, key="test", chunk_size=chunk_size)

        # Verify all key terms are preserved in chunks
        key_terms = ['ChromaDB', 'embeddings', 'SQLite', 'vectors', 'algorithms']
        all_text = ' '.join(c.text for c in chunks)
        for term in key_terms:
            assert term in all_text

        # Verify word boundaries are not squashed
        for chunk in chunks:
            assert 'queryembeddings' not in chunk.text
            assert 'APIfor' not in chunk.text
            assert 'searchalgorithms' not in chunk.text
            assert chunk.text.strip().endswith('.')


    @pytest.mark.parametrize('chunk_size', [5, 20, 50, 100, 1000])
    def test___chunk_text__exact_reconstruction(self, chunk_size: int):
        """Test that chunks exactly reconstruct the normalized text."""
        text = dedent("""
            First sentence goes here. Second sentence is a bit longer.
            Third sentence continues the text.    Fourth sentence has extra   spaces.

            Fifth sentence after newlines.

            Sixth sentence      with more spaces. Seventh sentence ends this test.
        """).strip()

        chunks = chunk_text(text, key="test", chunk_size=chunk_size)
        original_words = [w for w in text.split() if w]
        chunk_words = [w for chunk in chunks for w in chunk.text.split() if w]
        assert original_words == chunk_words
        # ensure each chunk ends with a sentence
        for chunk in chunks:
            assert chunk.text.strip().endswith('.')

    @pytest.mark.parametrize('chunk_size', [5, 20, 50, 100, 500])
    def test___chunk_text__sentence_endings(self, chunk_size: int):
        """Test handling of various sentence-ending punctuation."""
        text = dedent("""
            Question ended with space? Answer here.
            Exclamation with no space!Another sentence.
            Multiple dots... Then sentence.
            Spaces after dots...   Next sentence.
        """).strip()
        chunks = chunk_text(text, key="test", chunk_size=chunk_size)
        all_text = ' '.join(c.text for c in chunks)
        # Verify endings are preserved and spacing is normalized
        assert "space? Answer" in all_text
        assert "space!Another" in all_text
        assert "dots... Then" in all_text
        assert "dots... Next" in all_text

    @pytest.mark.parametrize('chunk_size', [5, 20, 50, 100])
    def test___chunk_text__special_characters(self, chunk_size: int):
        """Test handling of special characters and symbols."""
        text = "Sentence with §special© characters™. Another with μ∆∑ symbols."
        chunks = chunk_text(text, key="test", chunk_size=chunk_size)
        all_text = ' '.join(c.text for c in chunks)
        # Verify special characters are preserved
        assert all_text == text
