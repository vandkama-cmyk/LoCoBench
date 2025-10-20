"""
Adapter to use a small HuggingFace model (transformers pipeline) for generation.
This is intended for local testing with small models (e.g., google/flan-t5-small) and not optimized for large-scale production.
"""
from typing import Optional, List
import logging
import asyncio

from .rag import ModelAdapterInterface, RetrievedPassage

logger = logging.getLogger(__name__)


class HFLocalAdapter(ModelAdapterInterface):
    def __init__(self, model_name: str = 'google/flan-t5-small', device: Optional[int] = None, use_auth_token: Optional[str] = None):
        # device: -1 cpu, 0+ for GPU index
        self.model_name = model_name
        self.device = device if device is not None else -1
        self._pipe = None
        # Hugging Face auth token (string) or bool True to read from env HUGGINGFACE_TOKEN
        # If None, no auth token is passed.
        if use_auth_token is True:
            # try to read token from environment; if not present, attempt to load .env (optional)
            import os
            token = os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')
            if not token:
                # Try to load from a .env file in project root if python-dotenv is available
                try:
                    from pathlib import Path
                    from dotenv import load_dotenv

                    # project root assumed two levels up from this file
                    project_root = Path(__file__).resolve().parents[2]
                    env_path = project_root / '.env'
                    if env_path.exists():
                        load_dotenv(dotenv_path=env_path)
                        token = os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')
                        logger.info('Loaded HUGGINGFACE token from .env')
                except Exception:
                    # python-dotenv not installed or .env not present - ignore
                    logger.debug('python-dotenv not available or .env not found; attempting manual .env parse')
                    try:
                        # manual parse of .env
                        from pathlib import Path
                        project_root = Path(__file__).resolve().parents[2]
                        env_path = project_root / '.env'
                        if env_path.exists():
                            with open(env_path, 'r', encoding='utf-8') as f:
                                for line in f:
                                    line = line.strip()
                                    if not line or line.startswith('#'):
                                        continue
                                    if '=' in line:
                                        k, v = line.split('=', 1)
                                        k = k.strip()
                                        v = v.strip().strip('"')
                                        if k in ('HUGGINGFACE_TOKEN', 'HUGGINGFACE_HUB_TOKEN'):
                                            token = v
                                            logger.info('Loaded HUGGINGFACE token from .env (manual parse)')
                                            break
                    except Exception:
                        logger.debug('Manual .env parse failed or .env not present')

            self.use_auth_token = token
        else:
            self.use_auth_token = use_auth_token

    def _ensure_pipe(self):
        if self._pipe is None:
            # Use seq2seq pipeline for T5-like models
            try:
                # Import heavy HF libs lazily to avoid module import-time failures
                from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

                # Pass auth token if available (may be required for private models)
                if self.use_auth_token:
                    model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, use_auth_token=self.use_auth_token)
                    tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_auth_token=self.use_auth_token)
                else:
                    model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                    tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._pipe = pipeline('text2text-generation', model=model, tokenizer=tokenizer, device=self.device)
            except Exception as e:
                logger.error('Failed to load HF model %s: %s', self.model_name, e)
                raise

    async def generate(self, prompt: str, context: Optional[List[RetrievedPassage]] = None) -> str:
        # Ensure model pipeline is loaded (blocking load kept minimal)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._ensure_pipe)

        # Compose prompt (already includes RETRIEVED_CONTEXT when called from RAGClient)
        # For safety, we may truncate prompt if huge
        input_text = prompt if len(prompt) < 20000 else prompt[-20000:]

        # Use small generation params for speed
        out = await loop.run_in_executor(None, lambda: self._pipe(input_text, max_length=512, do_sample=False))
        if out and isinstance(out, list):
            return out[0].get('generated_text', '')
        return str(out)
 