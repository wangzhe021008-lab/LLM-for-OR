"""Download the public Chinese encoder to a project-local directory."""
import os
from pathlib import Path

root = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(root / "data/huggingface_cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "15")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
from sentence_transformers import SentenceTransformer

destination = root / "data/embedding_models/bge-small-zh-v1.5"
encoder = SentenceTransformer("BAAI/bge-small-zh-v1.5", device="cpu", cache_folder=str(root / "data/huggingface_cache"))
encoder.save(str(destination), safe_serialization=True)
vectors = encoder.encode(["整单配送成本不再乘客户需求", "电池容量与物理距离单位"], normalize_embeddings=True)
print({"model": "BAAI/bge-small-zh-v1.5", "local_path": str(destination), "dimensions": int(vectors.shape[1]), "device": str(encoder.device)})
