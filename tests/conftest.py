from smart_qa.client import LLMClient

@pytest.fixture
def client():
    """Fixture to provide an instance of LLMClient for tests."""
    return LLMClient()