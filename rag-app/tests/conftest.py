import pytest
from server.src.services.generation_service import generate_response
from unittest.mock import patch


@pytest.fixture
def mock_query():
    """Fixture to provide a sample query for testing."""
    return "Tell me about perovskites in solar cells."


@pytest.fixture
def mock_chunks():
    """Fixture to provide mock retrieved document chunks for generation tests."""
    return [
        {"chunk": "Perovskite materials are used in solar cells."},
        {"chunk": "Perovskites have unique electronic properties."},
        {"chunk": "The efficiency of perovskite solar cells has improved."},
    ]


@pytest.fixture
def mock_config():
    """Fixture for mock configuration settings."""
    return {
        "max_tokens": 150,
        "temperature": 0.7,
    }

@pytest.fixture
def mock_db_config():
    """Fixture for mock database configuration."""
    return {
        "dbname": "test_db",
        "user": "test_user",
        "password": "test_password",
        "host": "localhost",
        "port": "5432",
    }



@pytest.fixture
def mock_generate_response():
    """Fixture that mocks the LLM generation process in the generate_response function."""
    with patch("server.src.services.generation_service.generate_response") as mock_gen:
        yield mock_gen

@pytest.fixture
def mock_retrieve_top_k_chunks():
    """Fixture that mocks the Vector Database retrieval process
    in the retrieve_top_k_chunks function."""
    with patch("server.src.services.retrieval_service.retrieve_top_k_chunks") as mock_gen:
        yield mock_gen