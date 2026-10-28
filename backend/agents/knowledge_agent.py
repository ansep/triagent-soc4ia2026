from database.faiss_store import FAISSRuleStore
from config.llm_client import make_client
from config.settings import Provider

DEFAULT_MODEL = "llama-3.3-70b-versatile"


RULE_GENERATION_PROMPT = """You are an expert agricultural process analyst.
Given the following text from an agronomic document, extract a structured
process rule that an AI agent can follow to handle this situation.

Format the rule strictly as follows:
<<trigger>>: [one sentence describing when this rule applies]
<<steps>>:
1. [action step]
   <<decision>>: [condition]
     <<if yes>>: [action]
     <<if no>>: [action]
2. [next action step]
...
<<end>>

Rules:
- Be precise with dosages, product names, and timing from the document
- Preserve all decision gateways as explicit if/else conditions
- Include all relevant parameters (temperatures, thresholds, quantities)
- Do not invent information not present in the document

Document excerpt:
{text}

Generate the structured process rule:"""


RULE_GENERATION_UNSTRUCTURED_PROMPT = """You are an expert agricultural advisor.
Given the following situation description, generate a structured process rule
that an AI agent can follow to handle this type of situation.

Situation:
{text}

Generate the structured process rule:"""


class KnowledgeAgent:

    def __init__(self, rule_store: FAISSRuleStore, model: str = DEFAULT_MODEL,
                 provider: Provider = None):
        self.model = model
        self.client = make_client(provider)
        # vLLM: thinking off (as llama.cpp --reasoning off) and bounded output
        self._extra = ({"max_tokens": 2048,
                        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
                       if provider == "vllm" else {})
        self.rule_store = rule_store
        self.total_tokens = 0

    def generate_rule_from_document(self, text: str, metadata: dict = {}) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": RULE_GENERATION_PROMPT.format(text=text)}],
            temperature=0.0,
            **self._extra
        )
        self.total_tokens += resp.usage.total_tokens
        rule_content = resp.choices[0].message.content
        trigger = self._extract_trigger(rule_content)
        return self.rule_store.add_rule(trigger=trigger, content=rule_content, metadata=metadata)

    def generate_rule_from_description(self, description: str) -> dict:
        # no matching rule found: generate one on the fly and store it
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": RULE_GENERATION_UNSTRUCTURED_PROMPT.format(text=description)}],
            temperature=0.0,
            **self._extra
        )
        self.total_tokens += resp.usage.total_tokens
        rule_content = resp.choices[0].message.content
        trigger = self._extract_trigger(rule_content)
        rule_id = self.rule_store.add_rule(
            trigger=trigger,
            content=rule_content,
            metadata={"source": "on_the_fly", "original_description": description}
        )
        return {"rule_id": rule_id, "trigger": trigger, "content": rule_content, "similarity_score": 1.0}

    def handle_knowledge_request(self, description: str) -> dict:
        retrieved = self.rule_store.retrieve(description)
        if retrieved is not None:
            return {"source": "retrieved", **retrieved}
        generated = self.generate_rule_from_description(description)
        return {"source": "generated", **generated}

    def load_fao_documents(self, documents_path: str):
        import os
        try:
            import fitz
        except ImportError:
            print("install pymupdf: pip install pymupdf")
            return
        pdf_files = [f for f in os.listdir(documents_path) if f.endswith(".pdf")]
        print(f"found {len(pdf_files)} documents")
        for pdf_file in pdf_files:
            doc = fitz.open(os.path.join(documents_path, pdf_file))
            text = "".join(page.get_text() for page in doc)
            chunks = self._chunk_text(text)
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) > 200:
                    rid = self.generate_rule_from_document(
                        text=chunk,
                        metadata={"source": pdf_file, "chunk": i}
                    )
                    print(f"  rule {rid} <- {pdf_file} chunk {i}")
        print(f"rule store: {len(self.rule_store)} rules total")

    def _extract_trigger(self, rule_content: str) -> str:
        for line in rule_content.split("\n"):
            if "<<trigger>>" in line:
                return line.replace("<<trigger>>:", "").strip()
        return rule_content.split("\n")[0][:200]

    def _chunk_text(self, text: str, chunk_size: int = 1000) -> list:
        words = text.split()
        return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
