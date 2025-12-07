import pytest
import json
import os
from unittest.mock import Mock, MagicMock, patch, mock_open
from smart_qa.client import LLMClient, ExtractedEntities


class TestLLMClientInit:
    """Test the LLMClient initialization."""
    
    def test_init_missing_api_key(self):
        """Test that initialization raises ValueError when GEMINI_KEY is not set."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Gemini_API_KEY environment variable not set"):
                LLMClient()
    
    def test_init_with_api_key(self):
        """Test successful initialization with GEMINI_KEY set."""
        with patch.dict(os.environ, {"GEMINI_KEY": "test_key"}):
            with patch("smart_qa.client.genai.Client") as mock_client:
                client = LLMClient()
                assert client.client is not None
                mock_client.assert_called_once()


class TestSummarize:
    """Test the summarize method."""
    
    @patch.dict(os.environ, {"GEMINI_KEY": "test_key"})
    @patch("smart_qa.client.genai.Client")
    def test_summarize_returns_string(self, mock_genai_client):
        """Test that summarize returns a string."""
        # Setup mock
        mock_client_instance = MagicMock()
        mock_genai_client.return_value = mock_client_instance
        
        mock_response = MagicMock()
        mock_response.text = "This is a summary."
        mock_client_instance.models.generate_content.return_value = mock_response
        
        # Create client and call summarize
        client = LLMClient()
        result = client.summarize("Long text to summarize")
        
        assert isinstance(result, str)
        assert result == "This is a summary."
    
    @patch.dict(os.environ, {"GEMINI_KEY": "test_key"})
    @patch("smart_qa.client.genai.Client")
    def test_summarize_calls_cached_method(self, mock_genai_client):
        """Test that summarize calls the cached_summarize method."""
        mock_client_instance = MagicMock()
        mock_genai_client.return_value = mock_client_instance
        
        mock_response = MagicMock()
        mock_response.text = "Summary text"
        mock_client_instance.models.generate_content.return_value = mock_response
        
        client = LLMClient()
        
        with patch.object(LLMClient, "cached_summarize") as mock_cached:
            mock_cached.return_value = "Summary text"
            result = client.summarize("Text")
            mock_cached.assert_called_once()


class TestExtractEntities:
    """Test the extract_entities method."""
    
    @patch.dict(os.environ, {"GEMINI_KEY": "test_key"})
    @patch("smart_qa.client.genai.Client")
    def test_extract_entities_returns_dict(self, mock_genai_client):
        """Test that extract_entities returns a dictionary."""
        mock_client_instance = MagicMock()
        mock_genai_client.return_value = mock_client_instance
        
        entities_data = {
            "people": ["John Doe"],
            "dates": ["2023-01-01"],
            "locations": ["New York"],
            "unmatched_text": ""
        }
        
        mock_response = MagicMock()
        mock_response.text = json.dumps(entities_data)
        mock_client_instance.models.generate_content.return_value = mock_response
        
        client = LLMClient()
        result = client.extract_entities("John Doe visited New York on 2023-01-01.")
        
        assert isinstance(result, dict)
        assert result == entities_data
    
    @patch.dict(os.environ, {"GEMINI_KEY": "test_key"})
    @patch("smart_qa.client.genai.Client")
    def test_extract_entities_invalid_json(self, mock_genai_client):
        """Test that extract_entities raises error on invalid JSON response."""
        mock_client_instance = MagicMock()
        mock_genai_client.return_value = mock_client_instance
        
        mock_response = MagicMock()
        mock_response.text = "Invalid JSON {{"
        mock_client_instance.models.generate_content.return_value = mock_response
        
        client = LLMClient()
        with pytest.raises(json.JSONDecodeError):
            client.extract_entities("Some text")
    
    @patch.dict(os.environ, {"GEMINI_KEY": "test_key"})
    @patch("smart_qa.client.genai.Client")
    def test_extract_entities_system_prompt(self, mock_genai_client):
        """Test that extract_entities uses the correct system prompt."""
        mock_client_instance = MagicMock()
        mock_genai_client.return_value = mock_client_instance
        
        mock_response = MagicMock()
        mock_response.text = json.dumps({})
        mock_client_instance.models.generate_content.return_value = mock_response
        
        client = LLMClient()
        client.extract_entities("Test text")
        
        # Verify the API was called
        assert mock_client_instance.models.generate_content.called
        call_args = mock_client_instance.models.generate_content.call_args
        assert "NER" in call_args.kwargs["config"].system_instruction


class TestAsk:
    """Test the ask method."""
    
    @patch.dict(os.environ, {"GEMINI_KEY": "test_key"})
    @patch("smart_qa.client.genai.Client")
    def test_ask_returns_string(self, mock_genai_client):
        """Test that ask returns a string response."""
        mock_client_instance = MagicMock()
        mock_genai_client.return_value = mock_client_instance
        
        mock_response = MagicMock()
        mock_response.text = "The answer is 42."
        mock_client_instance.models.generate_content.return_value = mock_response
        
        client = LLMClient()
        result = client.ask("Context about something", "What is the answer?")
        
        assert isinstance(result, str)
        assert result == "The answer is 42."
    
    @patch.dict(os.environ, {"GEMINI_KEY": "test_key"})
    @patch("smart_qa.client.genai.Client")
    def test_ask_with_missing_info(self, mock_genai_client):
        """Test that ask can return 'Information not found' message."""
        mock_client_instance = MagicMock()
        mock_genai_client.return_value = mock_client_instance
        
        mock_response = MagicMock()
        mock_response.text = "Information not found in the provided context."
        mock_client_instance.models.generate_content.return_value = mock_response
        
        client = LLMClient()
        result = client.ask("Limited context", "Unrelated question")
        
        assert "Information not found" in result
    
    @patch.dict(os.environ, {"GEMINI_KEY": "test_key"})
    @patch("smart_qa.client.genai.Client")
    def test_ask_includes_context_and_question(self, mock_genai_client):
        """Test that ask includes both context and question in the prompt."""
        mock_client_instance = MagicMock()
        mock_genai_client.return_value = mock_client_instance
        
        mock_response = MagicMock()
        mock_response.text = "Answer"
        mock_client_instance.models.generate_content.return_value = mock_response
        
        client = LLMClient()
        context = "Important context"
        question = "Key question"
        client.ask(context, question)
        
        call_args = mock_client_instance.models.generate_content.call_args
        prompt = call_args.kwargs["contents"]
        assert context in prompt
        assert question in prompt


class TestReadTextFile:
    """Test the read_text_file method."""
    
    @patch.dict(os.environ, {"GEMINI_KEY": "test_key"})
    @patch("smart_qa.client.genai.Client")
    def test_read_text_file_success(self, mock_genai_client):
        """Test successful reading of a text file."""
        mock_genai_client.return_value = MagicMock()
        
        file_content = "This is the file content."
        with patch("builtins.open", mock_open(read_data=file_content)):
            client = LLMClient()
            result = client.read_text_file("test.txt")
            assert result == file_content
    
    @patch.dict(os.environ, {"GEMINI_KEY": "test_key"})
    @patch("smart_qa.client.genai.Client")
    def test_read_text_file_not_found(self, mock_genai_client):
        """Test reading a non-existent file raises FileNotFoundError."""
        mock_genai_client.return_value = MagicMock()
        
        client = LLMClient()
        with pytest.raises(FileNotFoundError):
            client.read_text_file("nonexistent.txt")
    
    @patch.dict(os.environ, {"GEMINI_KEY": "test_key"})
    @patch("smart_qa.client.genai.Client")
    def test_read_text_file_with_encoding(self, mock_genai_client):
        """Test that file is read with UTF-8 encoding."""
        mock_genai_client.return_value = MagicMock()
        
        file_content = "Content with special chars: é à ç"
        with patch("builtins.open", mock_open(read_data=file_content)):
            client = LLMClient()
            result = client.read_text_file("test.txt")
            assert result == file_content


class TestCachedSummarize:
    """Test the cached_summarize static method."""
    
    @patch.dict(os.environ, {"GEMINI_KEY": "test_key"})
    def test_cached_summarize_returns_string(self):
        """Test that cached_summarize returns a string."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Cached summary"
        mock_client.models.generate_content.return_value = mock_response
        
        result = LLMClient.cached_summarize(mock_client, "Long text")
        assert isinstance(result, str)
        assert result == "Cached summary"
    
    def test_cached_summarize_caches_results(self):
        """Test that cached_summarize caches results and doesn't call API twice."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Cached summary"
        mock_client.models.generate_content.return_value = mock_response
        
        # Call twice with same arguments
        result1 = LLMClient.cached_summarize(mock_client, "Same text")
        result2 = LLMClient.cached_summarize(mock_client, "Same text")
        
        # API should be called only once due to caching
        assert mock_client.models.generate_content.call_count == 1
        assert result1 == result2
    
    def test_cached_summarize_different_inputs_different_cache(self):
        """Test that different inputs result in different cache entries."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Summary"
        mock_client.models.generate_content.return_value = mock_response
        
        # Call with different text
        LLMClient.cached_summarize(mock_client, "Text one")
        LLMClient.cached_summarize(mock_client, "Text two")
        
        # API should be called twice for different inputs
        assert mock_client.models.generate_content.call_count == 2


class TestExtractedEntitiesModel:
    """Test the ExtractedEntities Pydantic model."""
    
    def test_extracted_entities_valid_data(self):
        """Test creating ExtractedEntities with valid data."""
        entities = ExtractedEntities(
            people=["John Doe", "Jane Smith"],
            dates=["2023-01-01"],
            locations=["New York", "London"],
            unmatched_text=""
        )
        assert len(entities.people) == 2
        assert len(entities.dates) == 1
        assert len(entities.locations) == 2
    
    def test_extracted_entities_empty_lists(self):
        """Test ExtractedEntities with empty lists."""
        entities = ExtractedEntities(
            people=[],
            dates=[],
            locations=[],
            unmatched_text="No entities found"
        )
        assert entities.people == []
        assert entities.unmatched_text == "No entities found"
    
    def test_extracted_entities_missing_field(self):
        """Test that ExtractedEntities requires all fields."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            ExtractedEntities(people=["John"])


class TestIntegration:
    """Integration tests for the LLMClient."""
    
    @patch.dict(os.environ, {"GEMINI_KEY": "test_key"})
    @patch("smart_qa.client.genai.Client")
    def test_full_workflow(self, mock_genai_client):
        """Test a complete workflow: read file, extract entities, ask question."""
        mock_client_instance = MagicMock()
        mock_genai_client.return_value = mock_client_instance
        
        # Mock responses for different methods
        extract_response = MagicMock()
        extract_response.text = json.dumps({
            "people": ["Alice"],
            "dates": ["2023-01-01"],
            "locations": ["Paris"],
            "unmatched_text": ""
        })
        
        ask_response = MagicMock()
        ask_response.text = "Alice is mentioned in the text."
        
        summarize_response = MagicMock()
        summarize_response.text = "A story about Alice in Paris."
        
        mock_client_instance.models.generate_content.side_effect = [
            extract_response, ask_response, summarize_response
        ]
        
        client = LLMClient()
        
        # Simulate workflow
        text = "Alice visited Paris on 2023-01-01."
        entities = client.extract_entities(text)
        answer = client.ask(text, "Who visited Paris?")
        summary = client.summarize(text)
        
        assert "Alice" in entities["people"]
        assert "Alice" in answer
        assert "story" in summary.lower()