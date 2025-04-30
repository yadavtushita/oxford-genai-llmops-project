import pytest
from server.src.services import retrieval_service

# Test function for the retrieval service - your postgres instance needs to be running.
@pytest.mark.asyncio
async def test_retrieve_top_k_chunks(mock_db_config, mock_retrieve_top_k_chunks):
    # Mock:
    query = "perovskite"
    top_k = 3
    mock_retrieve_top_k_chunks.return_value = [
        {"id": 1, "title": "Paper 1", "chunk": "Perovskite solar cells are efficient", "similarity_score": 0.1},
        {"id": 2, "title": "Paper 2", "chunk": "Perovskite materials in photovoltaics", "similarity_score": 0.2},
        {"id": 3, "title": "Paper 3", "chunk": "Advances in perovskite technology", "similarity_score": 0.3},
    ]

    # Call the function
    try:
        documents = retrieval_service.retrieve_top_k_chunks(query, top_k, mock_db_config)

        # Assertions
        assert isinstance(documents, list)
        assert len(documents) <= top_k

        for doc in documents:
            assert "id" in doc
            assert "title" in doc
            assert "chunk" in doc
            assert "similarity_score" in doc
    except Exception as e:
        pytest.fail(f"Test failed with error: {str(e)}")