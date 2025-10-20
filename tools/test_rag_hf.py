"""
Quick test script for RAG with sentence-transformers embeddings and a small HF model.
Run after installing requirements:

python tools/test_rag_hf.py

It will:
 - build an EmbeddingRetriever over a tiny in-memory project
 - load a small HF model (google/flan-t5-small) via HFLocalAdapter
 - run RAGClient.generate and print output
"""
import asyncio
import sys
from pathlib import Path

# Ensure project root is on sys.path so `locobench` package is importable when running the script
proj_root = Path(__file__).resolve().parents[1]
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

from locobench.utils.rag import RetrievedPassage, RAGClient
try:
    from locobench.utils.rag import EmbeddingRetriever, TfidfRetriever
except Exception:
    # fallback to TfidfRetriever if EmbeddingRetriever not available
    from locobench.utils.rag import TfidfRetriever

try:
    from locobench.utils.hf_model_adapter import HFLocalAdapter
except Exception:
    HFLocalAdapter = None


class MockAdapter:
    """Simple mock adapter that returns a canned response (used when HF/custom model not available)."""
    async def generate(self, prompt: str, context=None) -> str:
        return "MOCK_RESPONSE: This is a mocked generation used for testing RAG flow."


async def main():
    project_files = {
        "src/app.py": "def add(a, b):\n    return a + b\n\n# main app file\n",
        "src/utils.py": "def helper(x):\n    return x * 2\n",
        "README.md": "Sample project for RAG test: simple math utilities and an app."
    }

    # Try embedding retriever first; fall back to TF-IDF if sentence-transformers not installed or fails
    try:
        retriever = EmbeddingRetriever(project_files, model_name='all-MiniLM-L6-v2')
    except Exception as e:
        print('EmbeddingRetriever not available or failed:', e)
        retriever = TfidfRetriever(project_files)

    # Try to use HFLocalAdapter if available; otherwise use MockAdapter
    try:
        if HFLocalAdapter is not None:
            adapter = HFLocalAdapter(model_name='google/flan-t5-small', use_auth_token=True)
        else:
            raise RuntimeError('HFLocalAdapter not available')
    except Exception as e:
        print('HFLocalAdapter not available or failed:', e)
        adapter = MockAdapter()
    rag = RAGClient(retriever, adapter, max_context_chars=20000)

    prompt = "Create a task where a user must add a new feature to extend add() to support floats and update tests. Provide steps."
    out = await rag.generate(prompt, query='feature implementation', top_k=3)
    print('\n==== RAG OUTPUT ===\n', out)

if __name__ == '__main__':
    asyncio.run(main())
