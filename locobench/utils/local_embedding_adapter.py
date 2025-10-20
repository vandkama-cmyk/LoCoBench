from typing import Optional, List, Any
import logging

logger = logging.getLogger(__name__)


class LocalEmbeddingModel:
    """Light wrapper that loads a local embedding model.

    Usage:
      m = LocalEmbeddingModel(model_name_or_path='all-MiniLM-L6-v2')
      embs = m.encode(['text1','text2'])

    It will prefer sentence-transformers if available, otherwise try transformers
    with mean pooling over last hidden states.
    """

    def __init__(self, model_name_or_path: str = 'all-mpnet-base-v2', device: Optional[str] = None):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self._mode = None
        self._model = None
        self._tokenizer = None

        # Lazy detection
        try:
            import importlib
            if importlib.util.find_spec('sentence_transformers') is not None:
                self._mode = 'sentence-transformers'
            elif importlib.util.find_spec('transformers') is not None:
                self._mode = 'transformers'
            else:
                self._mode = None
        except Exception:
            self._mode = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        if self._mode == 'sentence-transformers':
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name_or_path)
                return
            except Exception as e:
                logger.warning('Failed to load sentence-transformers model: %s', e)
                self._model = None
        # fallback to transformers
        if self._mode == 'transformers' or self._model is None:
            try:
                from transformers import AutoTokenizer, AutoModel
                import torch
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
                self._model = AutoModel.from_pretrained(self.model_name_or_path)
                if self.device:
                    try:
                        self._model.to(self.device)
                    except Exception:
                        pass
            except Exception as e:
                logger.warning('Failed to load transformers model: %s', e)
                self._model = None

    def encode(self, texts: List[str], batch_size: int = 16):
        """Return numpy array of embeddings for texts."""
        self._ensure_loaded()
        if self._model is None:
            raise RuntimeError('No embedding model available (install sentence-transformers or transformers)')

        # sentence-transformers path
        try:
            if self._mode == 'sentence-transformers' and hasattr(self._model, 'encode'):
                return self._model.encode(texts, show_progress_bar=False)
        except Exception as e:
            logger.warning('sentence-transformers encode failed: %s', e)

        # transformers path: tokenize and mean-pool last hidden state
        try:
            import numpy as _np
            import torch
            all_embs = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                enc = self._tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
                if self.device:
                    try:
                        enc = {k: v.to(self.device) for k, v in enc.items()}
                    except Exception:
                        pass
                with torch.no_grad():
                    out = self._model(**enc)
                    last = out.last_hidden_state  # (bs, seq, dim)
                    mask = enc.get('attention_mask', None)
                    if mask is None:
                        emb = last.mean(dim=1).cpu().numpy()
                    else:
                        mask = mask.unsqueeze(-1)
                        summed = (last * mask).sum(dim=1)
                        denom = mask.sum(dim=1).clamp(min=1e-9)
                        emb = (summed / denom).cpu().numpy()
                    all_embs.append(emb)
            return _np.vstack(all_embs)
        except Exception as e:
            raise RuntimeError('Failed to encode with transformers model: %s' % e)
