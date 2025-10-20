"""
Adapter for a custom model served via HTTP. Keeps auth and request details configurable via environment or passed arguments.

This adapter implements the ModelAdapterInterface expected by RAGClient.
It sends JSON {"prompt": "..."} to the configured endpoint and expects a text response.
"""
import os
import aiohttp
import logging
from typing import List, Optional
from .rag import ModelAdapterInterface, RetrievedPassage

logger = logging.getLogger(__name__)


class CustomHTTPModelAdapter(ModelAdapterInterface):
    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None, timeout: int = 60):
        self.url = url or os.getenv('CUSTOM_MODEL_URL')
        self.api_key = api_key or os.getenv('CUSTOM_MODEL_API_KEY')
        self.timeout = timeout
        if not self.url:
            raise ValueError("Custom model URL not provided. Set CUSTOM_MODEL_URL or pass url param.")

    async def generate(self, prompt: str, context: Optional[List[RetrievedPassage]] = None) -> str:
        headers = {
            'Content-Type': 'application/json'
        }
        if self.api_key:
            headers['Authorization'] = f"Bearer {self.api_key}"

        payload = {
            'prompt': prompt,
            'max_tokens': 2000
        }

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            async with session.post(self.url, json=payload, headers=headers) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    logger.error("Custom model adapter received error status %s: %s", resp.status, text[:500])
                    raise RuntimeError(f"Custom model request failed: {resp.status} - {text[:200]}")
                return text