# TriAgent: repeated-run probes (SOC4IA 2026)

Code, instances and all runs behind the experiments of the SOC4IA 2026 position paper *How Do We Know Three Agents Are Enough?* (A. Sepielli, R. Grati, K. Boukadi, M. Napoleone, F. Monti, P. Merialdo, M. Mecella).

TriAgent is a three-agent architecture for crop-disease process automation, implemented with LangGraph:

- **Knowledge Agent**: retrieves a structured process rule from a FAISS store (all-MiniLM-L6-v2 embeddings);
- **Reasoning Agent**: runs a ReAct loop over the rule and decides each process gateway;
- **Execution Agent**: calls external tools (OpenWeatherMap weather, pest database, SMS).

The paper uses TriAgent as a running example of how architectural evidence is produced, not as a contribution in itself.

## What is in this repository

| Path | Content |
|---|---|
| `backend/` | TriAgent (agents, orchestrator, CLI used for all runs) |
| `data/rules/` | FAISS rule store and rules used at run time |
| `datasets/instances.jsonl` | 100 instances: 5 agronomic processes × 20 variants (SHA-256 in `instances.jsonl.sha256`) |
| `datasets/ground_truth.jsonl` | expected diagnosis, treatment, gateway decisions and tool calls per instance |
| `experiments/evaluate.py` | per-instance scoring |
| `experiments/analyze_probes.py` | probe analysis: per-seed accuracy, ICC(1), effective sample size, convergence, IQM |
| `infra/` | Docker runner and `run_probes.sh`, which drives every run |
| `results/probes/probeA/` | single-call baseline, temperature 0, 5 repetitions per model (1,000 runs) |
| `results/probes/probeB/` | Full vs No-Knowledge: 20 seeds on Gemma, 5 on Qwen (5,000 runs) |
| `results/probes/qwen_first_run_timeouts/` | first Qwen run, in which about half of the instances timed out (see below) |
| `results/analysis_output.txt` | output of `analyze_probes.py` on `results/probes` |

Each run folder contains `results.jsonl` (one line per instance, with the full reasoning trace) and `manifest.json` (seed, dataset hash, and the model actually served at the start and end of the batch).

## Setup

- Models: `nvidia/Gemma-4-26B-A4B-NVFP4` and `unsloth/Qwen3.8-27B-NVFP4`, each served by vLLM behind an OpenAI-compatible endpoint, thinking disabled.
- Sampling in the Reasoning Agent: the models' recommended settings (Gemma: T = 1.0, top-p 0.95, top-k 64; Qwen: T = 0.7, top-p 0.8, top-k 20, presence penalty 1.5). The seed is set per run through `TRIAGENT_SEED`.
- Runner: Docker on CPU (`infra/docker-compose.yml`); the GPU is used only by the vLLM servers.

## Reproducing the analysis

The analysis needs only the Python standard library:

```bash
python3 experiments/analyze_probes.py --selftest
python3 experiments/analyze_probes.py results/probes
```

The output matches `results/analysis_output.txt`.

## Re-running the probes

```bash
cp backend/.env.example backend/.env      # add an OpenWeatherMap key
export QWEN_URL=http://<host>:<port>/v1 GEMMA_URL=http://<host>:<port>/v1
bash infra/run_probes.sh build
bash infra/run_probes.sh check            # endpoints reachable and serving the expected models
bash infra/run_probes.sh A                # probe A
bash infra/run_probes.sh B                # probe B (hours); SEEDS="..." and ONLY=gemma4|qwen38 restrict it
bash infra/run_probes.sh analyze
```

Every leg uses `--resume`: rows that failed for infrastructure reasons (time-outs, unreachable server) are retried, successful rows are kept.

## Notes for interpreting the runs

- **Failures.** In the analysis, infrastructure errors are excluded and retried; agent errors (for example, a malformed reasoning step) count as incorrect. Four Gemma Full runs end with an agent error.
- **First Qwen run.** Without an output cap, some generations did not terminate and held the server until the client timed out; about half of the instances failed. On the surviving instances gateway accuracy was 0.65–0.69 per seed, against 0.33–0.37 for the complete rerun: the failures were the hard instances. The rerun used the current code, which caps each reasoning step at 2,048 tokens and disables thinking in the Knowledge Agent. The Gemma runs were made before this change.
- **Tool coverage.** The ground truth requires treatment tools (`get_treatment_protocol`, `calculate_dosage`, `check_waiting_period`) that the Reasoning Agent's system prompt does not list among its available tools; tool coverage is therefore 0 in all Full runs. The code is released as it was run.
- **Weather.** The Execution Agent queries live weather, so tool observations differ across runs and days.

## License

MIT (see `LICENSE`).
