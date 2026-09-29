"""RAG pipeline package.

Runs before any rag.* module, so library noise is silenced before those
libraries are imported. Both messages are harmless but print in red in
PowerShell and look like failures.
"""

import os
import warnings

# "You are sending unauthenticated requests to the HF Hub" -- models are public.
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
# Windows without Developer Mode can't symlink the HF cache; the fallback works fine.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# FAISS still lives in langchain-community, which now warns that it's being sunset.
warnings.filterwarnings("ignore", message=r".*langchain-community.*", category=DeprecationWarning)
