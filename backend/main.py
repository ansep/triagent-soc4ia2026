from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from orchestrator import TriAgentOrchestrator
from models.schemas import ProcessRequest, ProcessResult, HealthResponse
from config.settings import settings

orchestrators = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("starting up...")
    orchestrators["gpt"] = TriAgentOrchestrator(
        model="gpt-4o", provider="openai", rules_path=settings.rules_path)
    orchestrators["llama"] = TriAgentOrchestrator(
        model="llama-3.3-70b-versatile", provider="groq", rules_path=settings.rules_path)
    print(f"rules in store: {orchestrators['gpt'].get_rules_count()}")
    yield
    print("shutting down")


app = FastAPI(
    title="TriAgent",
    description="Three-layer multi-agent framework for agricultural process automation",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        rules_loaded=orchestrators["gpt"].get_rules_count(),
        llm_model="gpt-4o + llama-3.3-70b"
    )


@app.post("/process", response_model=ProcessResult)
async def process_request(request: ProcessRequest, model: str = "gpt"):
    if model not in orchestrators:
        raise HTTPException(status_code=400, detail=f"use 'gpt' or 'llama'")
    try:
        return orchestrators[model].process(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/load-documents")
async def load_documents(documents_path: str = "data/fao_documents", background_tasks: BackgroundTasks = None):
    def _load():
        orchestrators["gpt"].load_documents(documents_path)
    if background_tasks:
        background_tasks.add_task(_load)
        return {"message": "loading in background"}
    _load()
    return {"message": "done", "rules": orchestrators["gpt"].get_rules_count()}


@app.get("/rules")
async def list_rules():
    store = orchestrators["gpt"].rule_store
    return {
        "total": len(store),
        "rules": [
            {"rule_id": rid, "trigger": r["trigger"], "source": r["metadata"].get("source", "unknown")}
            for rid, r in store.rules.items()
        ]
    }


@app.get("/rules/{rule_id}")
async def get_rule(rule_id: str):
    store = orchestrators["gpt"].rule_store
    if rule_id not in store.rules:
        raise HTTPException(status_code=404, detail="not found")
    return store.rules[rule_id]


@app.get("/tools")
async def list_tools():
    return {"tools": orchestrators["gpt"].execution_agent.get_available_tools()}
