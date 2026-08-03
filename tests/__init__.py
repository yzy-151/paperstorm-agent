"""Test-suite defaults: force deterministic, offline-friendly settings
regardless of the outer shell environment."""

import os

os.environ["PAPERSTORM_RETRIEVAL_EMBEDDING"] = "hash"
os.environ["PAPERSTORM_CHAT_LLM"] = "0"
os.environ["PAPERSTORM_JUDGE_LLM"] = "0"
