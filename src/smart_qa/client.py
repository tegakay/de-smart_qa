import os
import json
import logging
from functools import lru_cache
# import google.generativeai as genai
from google import genai
from typing import Dict, Any, List
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="Utils INFO: %(message)s")

# from .custom_exceptions import LLMAPIError
from google.genai import types
from dotenv import load_dotenv
load_dotenv()


class LLMClient:
    def __init__(self) -> None:
        # TODO: Load API key and configure genai
        if not os.getenv("GEMINI_KEY"):
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        retry_options = types.HttpRetryOptions(
            attempts=5 
        )
        http_options = types.HttpOptions(
            timeout=180_000,           
            retry_options=retry_options
        ) 


        self.client = genai.Client(http_options=http_options)
        # self.summarize_cache = functools.lru_cache(maxsize=128)(self.summarize)
        


    def summarize(self, text: str) -> str:
        '''Summarizes the given text using the LLM.'''
        logger.info("Parent call")
        client = self.client

        return LLMClient.cached_summarize(client,text)

    
       

    def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extracts entities and returns a dictionary."""
        client = self.client

        system_prompt = (
        "You are an expert Named Entity Recognition (NER) system. "
        "Your only task is to extract the entities from the provided text and return them in the specific JSON format defined by the response schema. "
        "Do not add any commentary, prose, or markdown outside of the JSON object.")

        userprompt = f"Extract the entities from the following text:\n\n{text}"
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            content=userprompt,
            config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2
            )
        )
        return json.loads(response.text)



    def ask(self, context: str, question: str) -> str:
        """Answers a question based on the provided context."""
        system_prompt = (
            "You are an expert Question Answering system. "
            "Your task is to answer the user's question STRICTLY based on the provided CONTEXT. "
            "If the answer is not explicitly found in the context, you MUST respond with 'Information not found in the provided context.'"
        )

        user_prompt = f"CONTEXT:\n---\n{context}\n---\n\nQUESTION: {question}"
        client = self.client
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_prompt,
            config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2
            )
        )
        return response.text
        

    def read_text_file(self, file_path: str) -> str:
        """Reads and returns the content of a text file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    @staticmethod
    @lru_cache(maxsize=128)
    def cached_summarize(client,text: str) -> str:
        """A cached version of the summarize method."""
        logger.info("Making api call.")
        system_prompt = (
        "You are a helpful assistant that summarizes text concisely and accurately."
        )
        user_prompt = f"Please summarize the following text. Style: concise.\n\nText:\n{text}"


        response = client.models.generate_content(
        model="gemini-2.5-flash",  
        contents=user_prompt,
        config=types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.5,
        ))
        return response.text
        

class ExtractedEntities(BaseModel):
    """A collection of key entities extracted from the text."""
    
    # Use Field with a detailed description to guide the LLM
    people: List[str] = Field(
        description="A list of all unique full names of individuals mentioned in the text."
    )
    dates: List[str] = Field(
        description="A list of all unique dates, date ranges, or specific times mentioned (e.g., '1985', 'July 4, 1776', 'next Tuesday')."
    )
    locations: List[str] = Field(
        description="A list of all unique geographic locations mentioned (e.g., cities, countries, landmarks, buildings)."
    )
    unmatched_text: str = Field(
        description="If any remaining text contains significant information that does not fit into the other categories, place it here. Otherwise, use an empty string."
    )   

# llm = LLMClient()
# print(llm.summarize("what is the longest word in the english language without a vowel?"))