## Smart QA Project# Smart QA Project

A Python-based question answering and text processing system powered by Google's Gemini API. The project provides functionality for text summarization, entity extraction, and context-based question answering.

## Features

- **Text Summarization**: Summarize long texts concisely using Gemini
- **Entity Extraction**: Extract named entities (people, dates, locations) from text
- **Question Answering**: Answer questions based on provided context
- **File Reading**: Load and process text files
- **Caching**: Built-in caching for summarization to reduce API calls
- **Configurable**: HTTP timeouts and retry options for API calls

## Requirements

- Python >= 3.10
- Google Gemini API key
- Dependencies listed in `pyproject.toml`
- Poetry

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd smart_qa
```

2. Set Up Environment Variables
Create a .env file in the project root directory:

GEMINI_KEY=your_google_gemini_api_key_here

Replace your_google_gemini_api_key_here with your actual Google Gemini API key. You can obtain this from Google AI Studio.

install all dependencies using poetry install

## Usage
As a command line tool

```bash
poetry run smart_qa --file path/to/your/file.txt
```

this will read file.txt and print the summary

## Clear Cache

```bash
poetry run smart_qa --clear-cache
```

## As a python module
from smart_qa.client import LLMClient

# Initialize the client
client = LLMClient()

# Summarize text
text = "Your long text here..."
summary = client.summarize(text)
print(summary)

# Extract entities
entities = client.extract_entities(text)
print(entities)
# Output: {"people": [...], "dates": [...], "locations": [...], "unmatched_text": "..."}

# Ask a question based on context
answer = client.ask(context=text, question="What is the main topic?")
print(answer)

# Read from a file
file_content = client.read_text_file("path/to/file.txt")

# Clear the summarization cache
LLMClient.cached_summarize.cache_clear()