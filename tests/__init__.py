"""Test-suite defaults: keep retrieval deterministic and fast without models."""

import os

os.environ.setdefault("PAPERSTORM_RETRIEVAL_EMBEDDING", "hash")
