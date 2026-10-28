import os
import json
import uuid
import numpy as np
from typing import Optional

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("faiss not installed, falling back to numpy search")

EMBEDDING_MODEL = "text-embedding-3-small"
LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.75


class FAISSRuleStore:

    def __init__(self):
        self._api_key = os.getenv("OPENAI_API_KEY", "")
        # treat missing/placeholder keys as "no key" -> fall back to local embeddings
        self._use_local = not self._api_key.startswith("sk-")

        if self._use_local:
            from sentence_transformers import SentenceTransformer
            self._local_model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
            self._emb_dim = 384
        else:
            from openai import OpenAI
            self.client = OpenAI(api_key=self._api_key)
            self._emb_dim = 1536

        self.rules = {}
        self.embeddings = []
        self.rule_ids = []
        self.index = None
        self._init_index()

    def _init_index(self):
        # inner product on normalized vectors = cosine similarity
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(self._emb_dim)

    def _embed(self, text: str) -> np.ndarray:
        if self._use_local:
            vec = self._local_model.encode(text, convert_to_numpy=True).astype(np.float32)
        else:
            resp = self.client.embeddings.create(model=EMBEDDING_MODEL, input=text)
            vec = np.array(resp.data[0].embedding, dtype=np.float32)
        return vec / np.linalg.norm(vec)

    def add_rule(self, trigger: str, content: str, metadata: dict = {}) -> str:
        rule_id = str(uuid.uuid4())[:8]
        emb = self._embed(trigger)
        self.rules[rule_id] = {"trigger": trigger, "content": content, "metadata": metadata}
        self.embeddings.append(emb)
        self.rule_ids.append(rule_id)
        if FAISS_AVAILABLE:
            self.index.add(emb.reshape(1, -1))
        return rule_id

    def retrieve(self, query: str, top_k: int = 1) -> Optional[dict]:
        if not self.rule_ids:
            return None
        q = self._embed(query)
        if FAISS_AVAILABLE:
            scores, indices = self.index.search(q.reshape(1, -1), top_k)
            score = float(scores[0][0])
            idx = int(indices[0][0])
        else:
            mat = np.stack(self.embeddings)
            sims = mat @ q
            idx = int(np.argmax(sims))
            score = float(sims[idx])

        if score < SIMILARITY_THRESHOLD:
            return None

        rule_id = self.rule_ids[idx]
        rule = self.rules[rule_id]
        return {
            "rule_id": rule_id,
            "trigger": rule["trigger"],
            "content": rule["content"],
            "similarity_score": score,
            "metadata": rule["metadata"]
        }

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/rules.json", "w") as f:
            json.dump({
                "rules": self.rules,
                "rule_ids": self.rule_ids,
                "embeddings": [e.tolist() for e in self.embeddings]
            }, f)
        if FAISS_AVAILABLE and self.index.ntotal > 0:
            faiss.write_index(self.index, f"{path}/faiss.index")
        print(f"saved {len(self.rules)} rules to {path}")

    def load(self, path: str):
        rules_path = f"{path}/rules.json"
        if not os.path.exists(rules_path):
            return
        with open(rules_path) as f:
            data = json.load(f)
        self.rules = data["rules"]
        self.rule_ids = data["rule_ids"]
        self.embeddings = [np.array(e, dtype=np.float32) for e in data["embeddings"]]
        if FAISS_AVAILABLE:
            idx_path = f"{path}/faiss.index"
            if os.path.exists(idx_path):
                self.index = faiss.read_index(idx_path)
        print(f"loaded {len(self.rules)} rules")

    def __len__(self):
        return len(self.rules)
