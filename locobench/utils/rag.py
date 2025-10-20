"""
Lightweight Retrieval-Augmented Generation (RAG) utilities.
- TF-IDF retriever over project files
- RAGClient that combines retrieved passages with a model adapter
- Simple interfaces so custom adapters can be plugged in

Designed to be dependency-minimal. Uses scikit-learn for TF-IDF.
"""
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import math

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except Exception:
    TfidfVectorizer = None
    SKLEARN_AVAILABLE = False

try:
    # optional embedding retriever
    from sentence_transformers import SentenceTransformer
    import numpy as np
except Exception:
    SentenceTransformer = None
    np = None

logger = logging.getLogger(__name__)


@dataclass
class RetrievedPassage:
    file_path: str
    content: str
    score: float


class TfidfRetriever:
    """Simple TF-IDF retriever over an in-memory dict of file_path->content."""
    def __init__(self, documents: Dict[str, str], ngram_range=(1, 2)):
        self.docs = documents or {}
        self.file_paths = list(self.docs.keys())
        self.corpus = [self.docs[p] for p in self.file_paths]
        # If sklearn is available, use it; otherwise fallback to simple python implementation
        self.vectorizer = None
        self.doc_vectors = None
        if SKLEARN_AVAILABLE and self.corpus:
            try:
                self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=20000)
                self.doc_vectors = self.vectorizer.fit_transform(self.corpus)
                logger.info("TfidfRetriever (sklearn) initialized with %d documents", len(self.corpus))
            except Exception as e:
                logger.warning("Failed to initialize sklearn TfidfRetriever: %s", e)
                self.vectorizer = None
                self.doc_vectors = None
        else:
            # simple python fallback: precompute token sets and term frequencies
            self._tf_docs = []
            self._vocab = {}
            for doc in self.corpus:
                tokens = [t.lower() for t in _simple_tokenize(doc)]
                tf = {}
                for t in tokens:
                    tf[t] = tf.get(t, 0) + 1
                    if t not in self._vocab:
                        self._vocab[t] = len(self._vocab)
                self._tf_docs.append((tf, len(tokens)))
            logger.info("TfidfRetriever (python fallback) initialized with %d documents, vocab=%d", len(self.corpus), len(self._vocab))

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedPassage]:
        """Return top_k passages most relevant to query."""
        if not self.corpus or not query:
            return []

        if SKLEARN_AVAILABLE and self.vectorizer is not None and self.doc_vectors is not None:
            try:
                qv = self.vectorizer.transform([query])
                # compute cosine similarity
                import numpy as _np
                scores = (self.doc_vectors @ qv.T).toarray().ravel()
                ranked_idx = list(reversed(scores.argsort()))
                results: List[RetrievedPassage] = []
                for idx in ranked_idx[:top_k]:
                    score = float(scores[idx])
                    results.append(RetrievedPassage(self.file_paths[idx], self.corpus[idx], score))
                return results
            except Exception as e:
                logger.warning("TF-IDF (sklearn) retrieval failed: %s", e)
                return []
        else:
            # Simple python TF-IDF-like scoring: cosine between TF vectors (no idf)
            q_tokens = [t.lower() for t in _simple_tokenize(query)]
            q_tf = {}
            for t in q_tokens:
                q_tf[t] = q_tf.get(t, 0) + 1

            scores = []
            for i, (doc_tf, doc_len) in enumerate(self._tf_docs):
                # dot product
                dot = 0.0
                for t, v in q_tf.items():
                    dot += v * doc_tf.get(t, 0)
                # norm
                q_norm = math.sqrt(sum(v*v for v in q_tf.values()))
                d_norm = math.sqrt(sum(v*v for v in doc_tf.values()))
                sim = dot / (q_norm * d_norm + 1e-12)
                scores.append((i, sim))

            scores.sort(key=lambda x: x[1], reverse=True)
            results: List[RetrievedPassage] = []
            for idx, score in scores[:top_k]:
                results.append(RetrievedPassage(self.file_paths[idx], self.corpus[idx], float(score)))
            return results


def _simple_tokenize(text: str):
    # very small tokenizer: split on non-alphanumeric
    import re
    return [t for t in re.split(r"[^A-Za-z0-9_]+", text) if t]


class EmbeddingRetriever:
    """Retriever based on sentence-transformers embeddings.

    Pass a dict file_path->content and a sentence-transformer model name or instance.
    """

    def __init__(self, documents: Dict[str, str], model: Optional[Any] = None, model_name: str = 'all-mpnet-base-v2'):
        if SentenceTransformer is None:
            raise RuntimeError('sentence-transformers not available; install sentence-transformers')

        self.docs = documents or {}
        self.file_paths = list(self.docs.keys())
        self.corpus = [self.docs[p] for p in self.file_paths]
        # load or use provided model
        if model is not None:
            self.model = model
        else:
            self.model = SentenceTransformer(model_name)

        # compute embeddings
        try:
            self.embeddings = np.array(self.model.encode(self.corpus, show_progress_bar=False))
        except Exception as e:
            logger.warning('Failed to compute embeddings: %s', e)
            self.embeddings = None

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedPassage]:
        if not self.corpus or self.embeddings is None:
            return []

        q_emb = np.array(self.model.encode([query], show_progress_bar=False))[0]
        # cosine similarities
        sims = (self.embeddings @ q_emb) / (np.linalg.norm(self.embeddings, axis=1) * (np.linalg.norm(q_emb) + 1e-12))
        ranked_idx = list(reversed(sims.argsort()))
        results: List[RetrievedPassage] = []
        for idx in ranked_idx[:top_k]:
            score = float(sims[idx])
            results.append(RetrievedPassage(self.file_paths[idx], self.corpus[idx], score))
        return results


class ModelAdapterInterface:
    """Interface that model adapters should implement."""

    async def generate(self, prompt: str, context: Optional[List[RetrievedPassage]] = None) -> str:
        """Generate text given a prompt and optional retrieved context. Must be async."""
        raise NotImplementedError()


class RAGClient:
    """Retrieval-Augmented Generation client.

    Usage:
      retriever = TfidfRetriever(project_files)
      adapter = CustomAdapter(...)  # implements ModelAdapterInterface
      rag = RAGClient(retriever, adapter)
      output = await rag.generate(prompt)
    """

    def __init__(self, retriever: TfidfRetriever, adapter: ModelAdapterInterface, max_context_chars: int = 100000):
        self.retriever = retriever
        self.adapter = adapter
        self.max_context_chars = max_context_chars

    async def generate(self, prompt: str, query: Optional[str] = None, top_k: int = 5) -> str:
        # If user supplies a query for retrieval use it, otherwise use full prompt
        query_text = query or prompt
        retrieved = self.retriever.retrieve(query_text, top_k=top_k)

        # Truncate aggregated context to max_context_chars (simple heuristic)
        context_bundle: List[RetrievedPassage] = []
        total_chars = 0
        for p in retrieved:
            if total_chars + len(p.content) > self.max_context_chars:
                # Add truncated content
                remaining = max(0, self.max_context_chars - total_chars)
                if remaining <= 0:
                    break
                truncated = p.content[:remaining]
                context_bundle.append(RetrievedPassage(p.file_path, truncated, p.score))
                total_chars += len(truncated)
                break
            else:
                context_bundle.append(p)
                total_chars += len(p.content)

        logger.info("RAG retrieved %d passages, total_chars=%d", len(context_bundle), total_chars)

        # Build the augmented prompt passed to the adapter
        context_texts = []
        for p in context_bundle:
            header = f"FILE: {p.file_path} | SCORE: {p.score:.4f}\n"
            context_texts.append(header + p.content)

        augmented_prompt = "\n\n--- RETRIEVED CONTEXT ---\n\n".join(context_texts)
        if augmented_prompt:
            full_prompt = f"{prompt}\n\nRETRIEVED_CONTEXT:\n{augmented_prompt}\n\nPlease use the retrieved context to answer the prompt in detail." 
        else:
            full_prompt = prompt

        # Delegate to adapter
        return await self.adapter.generate(full_prompt, context=context_bundle)