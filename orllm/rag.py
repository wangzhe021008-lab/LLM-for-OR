"""Version-filtered retrieval: offline keyword baseline, optional Chroma vectors.

Keyword mode intentionally needs no third-party packages. Chroma mode uses
SentenceTransformer embeddings explicitly and never falls back to keywords.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re


def build_knowledge(root):
    """Load one complete logical rule per record; include current catalog rules."""
    root = Path(root)
    catalog = json.loads((root / "models/model_catalog.json").read_text(encoding="utf-8"))
    version = str(catalog["version"])
    records = []
    for i, model in enumerate(catalog["models"]):
        mid = model["model_id"]
        groups = {
            "scope": {"name": model["name"], "match_when": model["match_when"],
                      "assumptions": model["assumptions"]},
            "parameters": model["parameters"],
            "feasibility": model["feasibility_checks"],
        }
        for section, value in groups.items():
            records.append({"id": f"{mid}-{version}-catalog-{section}",
                "model_id": mid, "model_version": version,
                "title": f"{mid} {section}",
                "text": json.dumps(value, ensure_ascii=False),
                "source": f"models/model_catalog.json#/models/{i}",
                "category": section})
    for path in sorted((root / "knowledge").glob("*.jsonl")):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            required = {"id", "model_id", "model_version", "title", "text", "source", "category"}
            if not required <= record.keys():
                raise ValueError(f"Knowledge record incomplete: {path.name}:{line_no}")
            records.append(record)
    ids = [r["id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate knowledge record ids")
    registry=root/'knowledge/review_registry.json'
    if registry.is_file():
        approved=json.loads(registry.read_text(encoding='utf-8'))['records']
        records=[r for r in records if approved.get(r['id'],{}).get('status')=='engineering_reviewed'
                 and approved[r['id']].get('fingerprint')==_fingerprint([r])]
    return records


def _fingerprint(records):
    return hashlib.sha256(json.dumps(records, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_runtime_knowledge(root):
    """Read the reviewed workflow/data contract before intent planning."""
    root = Path(root)
    path = root / 'knowledge/runtime_contract.json'
    registry = json.loads((root / 'knowledge/review_registry.json').read_text(encoding='utf-8'))
    expected = registry.get('documents', {}).get('knowledge/runtime_contract.json')
    raw = path.read_bytes()
    if not expected or expected.get('status') != 'engineering_reviewed' or expected.get('sha256') != hashlib.sha256(raw).hexdigest():
        raise ValueError('运行知识文档尚未核对或内容发生变化，请更新工程核对记录。')
    return {'source': 'knowledge/runtime_contract.json', 'sha256': expected['sha256'],
            'document': json.loads(raw.decode('utf-8'))}


def retrieve_for_mapping(root, query, model_id, version='1.0', backend='model_pack', **options):
    """Mandatory complete pack first; similarity results can only supplement it."""
    required = retrieve(root, query, model_id, version=version, backend='model_pack', **options)
    supplemental = [] if backend == 'model_pack' else retrieve(root, query, model_id, version=version, backend=backend, **options)
    ids = {r['id'] for r in required}
    return [{**r, 'retrieval_role': 'required'} for r in required] + [
        {**r, 'retrieval_role': 'supplemental'} for r in supplemental if r['id'] not in ids]


def _tokens(text):
    text = text.lower()
    tokens = re.findall(r"[a-z0-9_]+", text)
    for fragment in re.findall(r"[\u3400-\u9fff]+", text):
        # Bigrams work for Chinese without downloading a segmenter dictionary.
        tokens.extend(fragment[j:j+2] for j in range(max(0, len(fragment)-1)))
        if len(fragment) == 1:
            tokens.append(fragment)
    return Counter(tokens)


def _keyword(query, records, top_k):
    q = _tokens(query)
    docs = [_tokens(r["title"] + " " + r["text"]) for r in records]
    df = Counter(token for doc in docs for token in doc)
    scored = []
    for record, doc in zip(records, docs):
        # A compact TF-IDF overlap score, not a probability or semantic score.
        score = sum((1 + math.log(doc[t])) * (1 + math.log((len(docs)+1)/(df[t]+1)))
                    for t in q if t in doc)
        score /= math.sqrt(max(1, sum(doc.values())))
        if score > 0:
            scored.append({**record, "backend": "keyword", "score": round(score, 8)})
    return sorted(scored, key=lambda r: (-r["score"], r["id"]))[:top_k]


def _index_path(root, index_path):
    path = Path(index_path) if index_path else Path("data/chroma")
    return path if path.is_absolute() else Path(root) / path


def _embedding_paths(root, embedding_model):
    """Keep bundled model identities relative when a workspace is relocated."""
    if not isinstance(embedding_model, str) or not embedding_model.strip():
        raise ValueError("Chroma mode requires an explicit embedding_model name or local path")
    root = Path(root).resolve()
    path = Path(embedding_model)
    candidate = path if path.is_absolute() else root / path
    if candidate.is_dir():
        candidate = candidate.resolve()
        identity = candidate.relative_to(root).as_posix() if candidate.is_relative_to(root) else str(candidate)
        return str(candidate), identity
    return embedding_model, embedding_model


def _vector_dependencies(embedding_model):
    if not isinstance(embedding_model, str) or not embedding_model.strip():
        raise ValueError("Chroma mode requires an explicit embedding_model name or local path")
    try:
        import chromadb
        from chromadb.config import Settings
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Chroma mode unavailable. Install requirements-rag.txt; no keyword fallback was used.") from exc
    encoder = SentenceTransformer(embedding_model)
    return chromadb, Settings, encoder


def build_index(root, embedding_model, index_path=None):
    """Build a persistent Chroma index. A model name may trigger model download."""
    records = build_knowledge(root)
    model_path, embedding_model = _embedding_paths(root, embedding_model)
    chromadb, Settings, encoder = _vector_dependencies(model_path)
    destination = _index_path(root, index_path)
    destination.mkdir(parents=True, exist_ok=True)
    corpus_hash = _fingerprint(records)
    signature = hashlib.sha256((corpus_hash + embedding_model).encode()).hexdigest()[:24]
    collection_name = "orllm_" + signature
    client = chromadb.PersistentClient(path=str(destination), settings=Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(name=collection_name, embedding_function=None,
        metadata={"corpus_hash": corpus_hash, "embedding_model": embedding_model})
    # Explicit embeddings avoid provider-specific embedding-function persistence.
    vectors = encoder.encode([r["title"] + "\n" + r["text"] for r in records],
        normalize_embeddings=True, show_progress_bar=False).tolist()
    for offset in range(0, len(records), 100):
        batch = records[offset:offset+100]
        collection.upsert(ids=[r["id"] for r in batch],
            documents=[r["text"] for r in batch], embeddings=vectors[offset:offset+100],
            metadatas=[{k: r[k] for k in ("model_id", "model_version", "title", "source", "category")} for r in batch])
    manifest = {"backend": "chroma", "embedding_model": embedding_model,
        "corpus_hash": corpus_hash, "collection": collection_name, "records": len(records)}
    (destination / "orllm_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**manifest, "index_path": str(destination)}


def retrieve(root, query, model_id, version="1.0", backend="keyword", top_k=5,
             embedding_model=None, index_path=None):
    if backend not in {"keyword", "chroma", "model_pack"}:
        raise ValueError("backend must be 'model_pack', 'keyword' or 'chroma'")
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    records = build_knowledge(root)
    filtered = [r for r in records if r["model_id"] == model_id and r["model_version"] == version]
    if not filtered:
        raise ValueError(f"No knowledge for selected model/version: {model_id}/{version}")
    if backend == 'model_pack':
        # Selection has already happened. Load all reviewed mapping rules for
        # this model; the complete catalog/schema are loaded directly by llm.py.
        root = Path(root)
        if not (root/'knowledge/review_registry.json').is_file():
            raise ValueError('模型知识包缺少核对清单，不能作为完整知识包加载。')
        manifest_path = root / 'knowledge/pack_manifest.json'
        manifest_raw = manifest_path.read_bytes()
        registry = json.loads((root/'knowledge/review_registry.json').read_text(encoding='utf-8'))
        approval = registry.get('documents', {}).get('knowledge/pack_manifest.json', {})
        if approval.get('status') != 'engineering_reviewed' or approval.get('sha256') != hashlib.sha256(manifest_raw).hexdigest():
            raise ValueError('知识包必需条目清单尚未核对或已经变化。')
        manifest = json.loads(manifest_raw.decode('utf-8'))
        expected = set(manifest['models'][model_id][version]['required_record_ids'])
        actual = {r['id'] for path in (root/'knowledge').glob('*.jsonl')
                  for line in path.read_text(encoding='utf-8').splitlines() if line.strip()
                  for r in [json.loads(line)] if r['model_id'] == model_id and r['model_version'] == version}
        if actual != expected:
            raise ValueError('残缺知识包或必需条目清单未同步，不能继续映射。')
        catalog_ids = set(manifest['models'][model_id][version]['catalog_record_ids'])
        if not catalog_ids <= {r['id'] for r in filtered}:
            raise ValueError('模型目录存在未核对或缺失的适用性、参数或可行性说明。')
        pack = [r for r in filtered if r['id'] in expected]
        if not expected or {r['id'] for r in pack} != expected:
            raise ValueError('模型知识包存在未经核对或内容已变化的条目，请先核对知识，不能加载残缺知识包。')
        return [{**r,'backend':'model_pack','review_status':'engineering_reviewed',
                 'content_sha256':_fingerprint([r])} for r in sorted(pack,key=lambda r:r['id'])]
    if backend == "keyword":
        return _keyword(query, filtered, top_k)
    destination = _index_path(root, index_path)
    manifest_path = destination / "orllm_manifest.json"
    if not manifest_path.exists():
        raise ValueError("Chroma index is missing. Run index-rag first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["corpus_hash"] != _fingerprint(records):
        raise ValueError("Knowledge/catalog changed: rebuild the Chroma index before retrieval")
    if embedding_model is None:
        embedding_model = manifest["embedding_model"]
    model_path, embedding_model = _embedding_paths(root, embedding_model)
    _, indexed_model = _embedding_paths(root, manifest["embedding_model"])
    if indexed_model != embedding_model:
        raise ValueError("Query embedding model differs from index embedding model: rebuild index")
    chromadb, Settings, encoder = _vector_dependencies(model_path)
    client = chromadb.PersistentClient(path=str(destination), settings=Settings(anonymized_telemetry=False))
    collection = client.get_collection(name=manifest["collection"], embedding_function=None)
    query_vector = encoder.encode([query], normalize_embeddings=True, show_progress_bar=False).tolist()
    result = collection.query(query_embeddings=query_vector, n_results=min(top_k, len(filtered)),
        where={"$and": [{"model_id": model_id}, {"model_version": version}]},
        include=["documents", "metadatas", "distances"])
    return [{"id": rid, "text": doc, **meta, "backend": "chroma", "distance": float(distance)}
        for rid, doc, meta, distance in zip(result["ids"][0], result["documents"][0],
            result["metadatas"][0], result["distances"][0])]
