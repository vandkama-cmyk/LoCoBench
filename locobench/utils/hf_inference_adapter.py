import os
from pathlib import Path
import asyncio
import logging
from typing import Optional, Dict, Any

try:
    from huggingface_hub import InferenceApi
except Exception:
    InferenceApi = None

from .hf_model_adapter import ModelAdapterInterface

logger = logging.getLogger(__name__)


class HFInferenceAdapter(ModelAdapterInterface):
    """Adapter that uses Hugging Face Inference API for text generation.

    Requires HUGGINGFACE_TOKEN in env or .env and an internet connection.
    This provides a lightweight fallback without installing torch/transformers.
    """

    def __init__(self, model: str = "jinaai/jina-embeddings-v2-base-code", use_auth_token: Optional[str] = None, task: str = 'text2text-generation'):
        self.model = model
        token = use_auth_token or os.environ.get('HUGGINGFACE_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN')
        # if token not in environment, try loading .env (best-effort)
        if token is None:
            try:
                # prefer python-dotenv if available
                from dotenv import load_dotenv
                load_dotenv()
                token = os.environ.get('HUGGINGFACE_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN')
            except Exception:
                # manual parse as fallback
                try:
                    env_path = Path('.') / '.env'
                    if env_path.exists():
                        for ln in env_path.read_text().splitlines():
                            if ln.strip().startswith('HUGGINGFACE_TOKEN='):
                                token = ln.split('=', 1)[1].strip()
                                break
                except Exception:
                    pass
        if token is None:
            logger.warning('No HUGGINGFACE_TOKEN found in env; Inference API will likely fail.')
        self.use_auth_token = token
        self._client = None
        self.task = task

    def _ensure_client(self):
        if self._client is not None:
            return
        if InferenceApi is None:
            raise RuntimeError('huggingface_hub not installed; please `pip install huggingface-hub`')
        # pass explicit task to help the Inference API route the request
        self._client = InferenceApi(repo_id=self.model, token=self.use_auth_token, task=self.task)

    async def generate(self, prompt: str, max_tokens: int = 256, context: Optional[Dict[str, Any]] = None) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._generate_sync, prompt, max_tokens, context)

    def _generate_sync(self, prompt: str, max_tokens: int = 256, context: Optional[Dict[str, Any]] = None) -> str:
        self._ensure_client()
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens}}
        if context:
            # include context as part of prompt if provided
            ctx_text = ''
            if isinstance(context, dict):
                # try to extract passages
                passages = context.get('passages') or context.get('documents') or []
                if isinstance(passages, (list, tuple)):
                    ctx_text = '\n\n'.join(p.get('text', str(p)) if isinstance(p, dict) else str(p) for p in passages)
            if ctx_text:
                payload['inputs'] = f"Context:\n{ctx_text}\n\nPrompt:\n{prompt}"

        # Request raw response to support text/plain outputs from some models
        try:
            res = self._client(payload, raw_response=True)
        except TypeError:
            # older versions may not accept raw_response; fall back
            res = self._client(payload)

        # If we got a requests.Response-like object, extract text
        try:
            # duck-typing: requests.Response has .status_code and .text
            status = getattr(res, 'status_code', None)
            text = getattr(res, 'text', None)
            if text is not None:
                return text
        except Exception:
            pass

        # response may be a dict or list depending on model
        if isinstance(res, dict) and 'generated_text' in res:
            return res['generated_text']
        if isinstance(res, list) and len(res) > 0:
            # some Inference API models return list of {generated_text: ...}
            first = res[0]
            if isinstance(first, dict) and 'generated_text' in first:
                return first['generated_text']
            # otherwise try text field
            if isinstance(first, dict) and 'text' in first:
                return first['text']
            return str(first)
        # fallback: string or other
        return str(res)
