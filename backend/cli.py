"""
Command-line interface for running TriAgent batch experiments.

Usage:
    python -m cli batch --input datasets/instances.jsonl \
        --output results/smoke/gpt-4o__full/results.jsonl \
        --backbone gpt-4o --provider openai --ablation full

The output directory is created automatically.
Each line in the output JSONL has the schema:
    {run_id, instance_id, backbone, provider, ablation, latency_s,
     result: ProcessResult | null, error: str | null}
A manifest.json is written alongside results.jsonl on completion.
"""

import json
import time
import uuid
import hashlib
import os
from pathlib import Path
from datetime import datetime, timezone

import click
import requests

from orchestrator import TriAgentOrchestrator
from models.schemas import ProcessRequest
from config.settings import settings


@click.group()
def cli():
    pass


@cli.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True),
              help="JSONL file with test instances (datasets/instances.jsonl)")
@click.option("--output", "output_path", required=True, type=click.Path(),
              help="Output JSONL path, e.g. results/probes/probeB/gemma4__full__seed11/results.jsonl")
@click.option("--backbone", default=None,
              help="LLM model name, e.g. gpt-4o, gemma4-local. Defaults to TRIAGENT_BACKBONE env.")
@click.option("--provider", default=None,
              type=click.Choice(["openai", "groq", "deepseek", "llamacpp", "vllm"]),
              help="Provider for the backbone. Defaults to TRIAGENT_BACKBONE_PROVIDER env.")
@click.option("--ablation", default="full",
              type=click.Choice(["full", "no_knowledge", "no_execution"]),
              help="Ablation mode.")
@click.option("--run-id", default=None,
              help="Optional run ID for tracing. Auto-generated if not set.")
@click.option("--limit", type=int, default=None,
              help="Stop after N instances (useful for smoke tests).")
@click.option("--resume", is_flag=True, default=False,
              help="Skip instances whose instance_id already appears in the output file.")
@click.option("--skip-prefetch", "skip_prefetch", is_flag=True, default=False,
              help="Skip the weather pre-fetch so models must request get_weather via requires_tool.")
def batch(input_path, output_path, backbone, provider, ablation, run_id, limit, resume, skip_prefetch):
    backbone = backbone or settings.backbone
    provider = provider or settings.backbone_provider
    run_id = run_id or str(uuid.uuid4())[:8]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # collect already-processed IDs for --resume
    done_ids: set[str] = set()
    if resume and out.exists():
        with open(out) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if not row.get("error"):  # failed rows (e.g. a time-out) get retried
                        done_ids.add(row["instance_id"])
                except Exception:
                    pass
        click.echo(f"resuming: {len(done_ids)} instances already done")

    # dataset hash for manifest
    dataset_sha = _file_sha256(input_path)

    orch = TriAgentOrchestrator(
        model=backbone,
        provider=provider,
        rules_path=settings.rules_path,
        ablation_mode=ablation,
        skip_prefetch=skip_prefetch,
    )

    served_start = _served_model(provider)
    started_at = datetime.now(timezone.utc).isoformat()
    processed = 0
    failed = 0

    with open(input_path) as fin, open(out, "a") as fout:
        for line in fin:
            if limit and processed >= limit:
                break

            inst = json.loads(line.strip())
            iid = inst["instance_id"]

            if iid in done_ids:
                continue

            t0 = time.time()

            try:
                req = ProcessRequest(**inst["request"])
                res = orch.process(req)
                row = {
                    "run_id": run_id,
                    "instance_id": iid,
                    "process_id": inst.get("process_id"),
                    "backbone": backbone,
                    "provider": provider,
                    "ablation": ablation,
                    "latency_s": round(time.time() - t0, 2),
                    "result": res.model_dump(),
                    "error": None,
                }
            except Exception as e:
                failed += 1
                row = {
                    "run_id": run_id,
                    "instance_id": iid,
                    "process_id": inst.get("process_id"),
                    "backbone": backbone,
                    "provider": provider,
                    "ablation": ablation,
                    "latency_s": round(time.time() - t0, 2),
                    "result": None,
                    "error": str(e),
                }

            fout.write(json.dumps(row) + "\n")
            fout.flush()
            processed += 1

    ended_at = datetime.now(timezone.utc).isoformat()

    manifest = {
        "run_id": run_id,
        "backbone": backbone,
        "provider": provider,
        "ablation": ablation,
        "started_at": started_at,
        "ended_at": ended_at,
        "total_processed": processed,
        "total_failed": failed,
        "dataset_sha256": dataset_sha,
        "input_path": str(input_path),
        "seed": os.environ.get("TRIAGENT_SEED"),
        "served_model_start": served_start,
        "served_model_end": _served_model(provider),
    }
    manifest_path = out.parent / "manifest.json"
    with open(manifest_path, "w") as mf:
        json.dump(manifest, mf, indent=2)

    click.echo(f"done: {processed} instances, {failed} errors → {out}")
    click.echo(f"manifest: {manifest_path}")


_MONO_SYSTEM = (
    "You are an expert agricultural advisor. "
    "Given a field situation description, identify the disease or pest and recommend treatment.\n\n"
    "Respond in exactly this format (two lines, nothing else):\n"
    "DIAGNOSIS: <pathogen name and disease>\n"
    "TREATMENT: <active ingredient(s) with dosage in L/ha or g/ha and timing>"
)

_MONO_USER = (
    "Field situation:\n{description}\n\n"
    "Crop: {crop}\nLocation: {location}\nGrowth stage: {growth_stage}"
)


@cli.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True),
              help="JSONL file with test instances")
@click.option("--output", "output_path", required=True, type=click.Path(),
              help="Output JSONL path")
@click.option("--backbone", default=None)
@click.option("--provider", default=None,
              type=click.Choice(["openai", "groq", "deepseek", "llamacpp", "vllm"]))
@click.option("--limit", type=int, default=None)
@click.option("--resume", is_flag=True, default=False)
def monolithic(input_path, output_path, backbone, provider, limit, resume):
    """Single-shot LLM baseline: no agents, no FAISS, no ReAct. One prompt per instance."""
    from config.llm_client import make_client

    backbone = backbone or settings.backbone
    provider = provider or settings.backbone_provider
    run_id = str(uuid.uuid4())[:8]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    if resume and out.exists():
        with open(out) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if not row.get("error"):  # failed rows (e.g. a time-out) get retried
                        done_ids.add(row["instance_id"])
                except Exception:
                    pass
        click.echo(f"resuming: {len(done_ids)} already done")

    client = make_client(provider)
    served_start = _served_model(provider)
    started_at = datetime.now(timezone.utc).isoformat()
    processed = 0
    failed = 0

    with open(input_path) as fin, open(out, "a") as fout:
        for line in fin:
            if limit and processed >= limit:
                break
            inst = json.loads(line.strip())
            iid = inst["instance_id"]
            if iid in done_ids:
                continue

            req = inst["request"]
            t0 = time.time()
            try:
                user_msg = _MONO_USER.format(
                    description=req["description"],
                    crop=req.get("crop") or "unknown",
                    location=req.get("location") or "unknown",
                    growth_stage=req.get("growth_stage") or "unknown",
                )
                resp = client.chat.completions.create(
                    model=backbone,
                    messages=[
                        {"role": "system", "content": _MONO_SYSTEM},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.0,
                    max_tokens=256,
                    # vLLM serves Qwen3/Gemma4 with thinking on by default; the agents run with it off
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}}
                    if provider == "vllm" else None,
                )
                raw = resp.choices[0].message.content or ""
                latency = round(time.time() - t0, 2)

                # parse DIAGNOSIS / TREATMENT lines
                diagnosis = ""
                treatment = ""
                for ln in raw.splitlines():
                    ln = ln.strip()
                    if ln.upper().startswith("DIAGNOSIS:"):
                        diagnosis = ln.split(":", 1)[1].strip()
                    elif ln.upper().startswith("TREATMENT:"):
                        treatment = ln.split(":", 1)[1].strip()
                if not diagnosis:
                    diagnosis = raw[:200]
                if not treatment:
                    treatment = raw

                row = {
                    "run_id": run_id,
                    "instance_id": iid,
                    "process_id": inst.get("process_id"),
                    "backbone": backbone,
                    "provider": provider,
                    "ablation": "monolithic",
                    "latency_s": latency,
                    "result": {
                        "success": True,
                        "process_id": iid,
                        "diagnosis": diagnosis,
                        "treatment": treatment,
                        "actions_taken": [],
                        "reasoning_trace": [],
                        "rule_used": None,
                        "hallucination_flags": [],
                        "escalated": False,
                        "cost_usd": 0.0,
                        "llm_model": backbone,
                    },
                    "error": None,
                }
            except Exception as e:
                failed += 1
                row = {
                    "run_id": run_id,
                    "instance_id": iid,
                    "process_id": inst.get("process_id"),
                    "backbone": backbone,
                    "provider": provider,
                    "ablation": "monolithic",
                    "latency_s": round(time.time() - t0, 2),
                    "result": None,
                    "error": str(e),
                }

            fout.write(json.dumps(row) + "\n")
            fout.flush()
            processed += 1

    ended_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "run_id": run_id, "backbone": backbone, "provider": provider,
        "ablation": "monolithic", "started_at": started_at, "ended_at": ended_at,
        "total_processed": processed, "total_failed": failed,
        "input_path": str(input_path),
        "served_model_start": served_start,
        "served_model_end": _served_model(provider),
    }
    with open(out.parent / "manifest.json", "w") as mf:
        json.dump(manifest, mf, indent=2)

    click.echo(f"done: {processed} instances, {failed} errors -> {out}")


@cli.command()
@click.option("--rules", "rules_path", required=True, type=click.Path(exists=True),
              help="JSONL file with seed rules (datasets/seed_rules.jsonl)")
@click.option("--rules-path", "store_path", default=None,
              help="Path to the FAISS rule store directory. Defaults to settings.rules_path.")
def seed(rules_path, store_path):
    """Pre-seed the FAISS rule store with canonical process rules.

    Each line in the rules JSONL must have: rule_id, trigger, content, source.
    Run this before any 'full' ablation batch to avoid cold-start retrieval failures.
    """
    from database.faiss_store import FAISSRuleStore

    store_dir = store_path or settings.rules_path
    store = FAISSRuleStore()
    store.load(store_dir)

    with open(rules_path) as f:
        rules = [json.loads(line) for line in f if line.strip()]

    for rule in rules:
        meta = rule.get("metadata", {})
        meta["source"] = rule.get("source", "manual_seed")
        meta["process_id"] = rule.get("process_id", "")
        store.add_rule(
            trigger=rule["trigger"],
            content=rule["content"],
            metadata=meta,
        )

    store.save(store_dir)
    click.echo(f"seeded {len(rules)} rules into {store_dir}")


def _served_model(provider: str) -> str | None:
    """What the local server actually serves: the GGUF loaded by llama.cpp (/props; it
    ignores the `model` field we send) or the model id exposed by vLLM (/v1/models)."""
    try:
        if provider == "llamacpp":
            base = settings.llamacpp_base_url.rsplit("/v1", 1)[0]
            return requests.get(f"{base}/props", timeout=5).json().get("model_path")
        if provider == "vllm":
            data = requests.get(f"{settings.vllm_base_url}/models", timeout=5).json()["data"]
            return ", ".join(m.get("root") or m["id"] for m in data)
    except Exception as e:
        return f"unavailable: {e}"
    return None


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    cli()
