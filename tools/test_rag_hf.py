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
        from locobench.utils.local_embedding_adapter import LocalEmbeddingModel
    except Exception:
        LocalEmbeddingModel = None

try:
    from locobench.utils.hf_model_adapter import HFLocalAdapter
except Exception:
    HFLocalAdapter = None
try:
    from locobench.utils.hf_inference_adapter import HFInferenceAdapter
except Exception:
    HFInferenceAdapter = None


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
        # if user provided LOCAL_EMBED_MODEL env var, use LocalEmbeddingModel
        import os
        local_model_path = os.environ.get('LOCAL_EMBED_MODEL')
        if local_model_path and LocalEmbeddingModel is not None:
            model_instance = LocalEmbeddingModel(model_name_or_path=local_model_path)
            retriever = EmbeddingRetriever(project_files, model=model_instance)
        else:
            retriever = EmbeddingRetriever(project_files, model_name='all-MiniLM-L6-v2')
    except Exception as e:
        print('EmbeddingRetriever not available or failed:', e)
        retriever = TfidfRetriever(project_files)

    # Try to use HFLocalAdapter if available; otherwise fallback to HFInferenceAdapter or MockAdapter
    adapter = None
    # Prefer the lightweight Inference API adapter to avoid importing heavy local libs
    if HFInferenceAdapter is not None:
        # Try a small set of public models/tasks and pick the first that responds
        candidate_models = [
            ('distilgpt2', 'text-generation'),
            ('gpt2', 'text-generation'),
            ('sshleifer/distilbart-cnn-12-6', 'summarization')
        ]
        for mid, task in candidate_models:
            try:
                cand = HFInferenceAdapter(model=mid, use_auth_token=True, task=task)
                # quick health check
                try:
                    # perform a health check by making a simple inference call
                    resp = cand.generate('Hello world')
                    continue
                except Exception as e:
                    print(f'model {mid} failed check: {e}')
                    continue
                print('Using HFInferenceAdapter with model', mid, 'task', task)
                adapter = cand
                break
            except Exception as e:
                print('HFInferenceAdapter init failed for', mid, e)
                adapter = None

    # Optionally use local HF if explicitly requested and imports look safe
    use_local = False
    try:
        import os
        use_local = os.environ.get('USE_LOCAL_HF', '') == '1'
    except Exception:
        use_local = False

    if adapter is None and use_local and HFLocalAdapter is not None:
        try:
            # quick import check for transformers to avoid raising heavy import errors
            import importlib
            importlib.import_module('transformers')
            adapter = HFLocalAdapter(model_name='google/flan-t5-small', use_auth_token=True)
            print('Using HFLocalAdapter (local transformers)')
        except Exception as e:
            print('HFLocalAdapter not available or unsafe to import:', e)
            adapter = None

    if adapter is None:
        print('Falling back to MockAdapter for generation')
        adapter = MockAdapter()
    rag = RAGClient(retriever, adapter, max_context_chars=20000)

    prompt = "Create a task where a user must add a new feature to extend add() to support floats and update tests. Provide steps."
    out = await rag.generate(prompt, query='feature implementation', top_k=3)
    print('\n==== RAG OUTPUT ===\n', out)

if __name__ == '__main__':
    asyncio.run(main())
