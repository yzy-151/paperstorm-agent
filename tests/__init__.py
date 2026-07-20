"""Test-suite defaults: keep retrieval deterministic and fast without models."""

import os

os.environ.setdefault("PAPERSTORM_RETRIEVAL_EMBEDDING", "hash")
os.environ.setdefault("PAPERSTORM_CHAT_LLM", "0")
os.environ.setdefault("PAPERSTORM_JUDGE_LLM", "0")
