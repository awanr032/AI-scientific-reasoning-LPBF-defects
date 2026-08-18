from lpbf_defect_reasoning.indexing import SemanticIndex


def test_semantic_index_len_matches_chunk_count(sample_chunks, fake_embedder):
    index = SemanticIndex(sample_chunks, fake_embedder)
    assert len(index) == len(sample_chunks)


def test_semantic_index_search_returns_top_k(sample_chunks, fake_embedder):
    index = SemanticIndex(sample_chunks, fake_embedder)
    results = index.search("keyhole porosity", top_k=2)

    assert len(results) == 2
    assert all("chunk_id" in r for r in results)


def test_semantic_index_search_finds_lexically_similar_chunk(sample_chunks, fake_embedder):
    index = SemanticIndex(sample_chunks, fake_embedder)
    # c1's text shares many tokens with this query under the bag-of-words fake embedder.
    results = index.search("High laser power increases the risk of keyhole porosity.", top_k=1)
    assert results[0]["chunk_id"] == "c1"
