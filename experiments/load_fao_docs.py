"""
Load PDF documents from a directory into the FAISS rule store via KnowledgeAgent.
Usage: python experiments/load_fao_docs.py [docs_path] [rules_store_path]
Defaults: docs_path=data/fao_documents  rules_store_path=data/rules
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents.knowledge_agent import KnowledgeAgent
from database.faiss_store import FAISSRuleStore

docs_path   = sys.argv[1] if len(sys.argv) > 1 else "data/fao_documents"
store_path  = sys.argv[2] if len(sys.argv) > 2 else "data/rules"

if not os.path.isdir(docs_path):
    print(f"ERROR: directory not found: {docs_path}")
    sys.exit(1)

pdfs = [f for f in os.listdir(docs_path) if f.lower().endswith(".pdf")]
if not pdfs:
    print(f"ERROR: no PDF files found in {docs_path}")
    sys.exit(1)

print(f"Found {len(pdfs)} PDF(s): {pdfs}")

store = FAISSRuleStore()
store.load(store_path)
print(f"Rule store loaded: {len(store.rules)} existing rules")

ka = KnowledgeAgent(
    rule_store=store,
    model="deepseek-chat",
    provider="deepseek",
)

ka.load_fao_documents(docs_path)

store.save(store_path)
print(f"Done. Rule store now has {len(store.rules)} rules. Saved to {store_path}")
