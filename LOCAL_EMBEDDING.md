# Using a Local Embedding Model with LoCoBench RAG

This guide explains how to use a local embedding model (either a sentence-transformers model or a transformers model directory) with the LoCoBench RAG utilities.

Overview
- `locobench/utils/local_embedding_adapter.py` contains `LocalEmbeddingModel` that wraps local models and exposes `.encode(texts)`.
- `locobench/utils/rag.py`'s `EmbeddingRetriever` now accepts a `model` instance with `.encode()` or a `model_name` when `sentence-transformers` is available.
- `tools/test_rag_hf.py` can use a local embedding model by setting the `LOCAL_EMBED_MODEL` environment variable.

Options to run locally

1) Use a sentence-transformers model (recommended)

- Install (CPU-only):

```bash
python -m pip install --upgrade pip
python -m pip install "sentence-transformers==2.2.2" scikit-learn
```

- Run the test harness with a HF model name (will download if not cached):

```bash
export LOCAL_EMBED_MODEL="all-MiniLM-L6-v2"
python tools/test_rag_hf.py
```

2) Use a local transformers model directory (fallback)

- If you have a transformers model saved locally or downloaded via `huggingface-cli`:

```bash
export LOCAL_EMBED_MODEL="/path/to/local/model"
python tools/test_rag_hf.py
```

- Ensure you have transformers & torch installed (CPU-only):

```bash
python -m pip install transformers torch
```

3) If you can't install heavy libs locally

- Use the Hugging Face Inference API (requires a token in `.env` or `HUGGINGFACE_TOKEN` in env). The test harness will automatically use the HF Inference Adapter for generation and can be used for basic testing.

Notes & tips

- The `LocalEmbeddingModel` will prefer `sentence-transformers` if present because it provides a convenient `encode()` method; otherwise it uses `transformers` + mean pooling.
- If you set `LOCAL_EMBED_MODEL`, `tools/test_rag_hf.py` will create a LocalEmbeddingModel and pass it to `EmbeddingRetriever`. Otherwise, it will try the default `all-MiniLM-L6-v2` via sentence-transformers.
- If you run into numpy or binary compatibility errors on Windows, consider creating a fresh venv to avoid conflicts.

Example (fresh venv recommended):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install sentence-transformers scikit-learn
export LOCAL_EMBED_MODEL="all-MiniLM-L6-v2"
python tools/test_rag_hf.py
```

If you'd like, I can also add a small unit test that exercises `EmbeddingRetriever` with a `LocalEmbeddingModel` to ensure the encoding and retrieval end-to-end works on machines with the required deps.
