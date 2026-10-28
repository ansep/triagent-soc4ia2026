#!/bin/bash
# Repeated-run probes of the SOC4IA 2026 paper. The runner works inside Docker
# (infra/docker-compose.yml); the two models are served by vLLM (OpenAI-compatible API).
# Host requirements: docker, bash, curl.
#
#   bash infra/run_probes.sh build     # build the runner image (once)
#   bash infra/run_probes.sh check     # endpoints reachable and serving the expected models
#   bash infra/run_probes.sh smoke     # 2 instances x 2 runs per model with the same seed
#   bash infra/run_probes.sh A         # single-call baseline at T=0, 5 repetitions per model
#   nohup bash infra/run_probes.sh B > probes_B.log 2>&1 &   # full vs no_knowledge, 5 seeds, both models
#   bash infra/run_probes.sh analyze   # summary of A and B
#
# B runs the 5 seeds of each model concurrently (vLLM batches them); full and
# no_knowledge share the seed. Restartable: every leg uses --resume.
# Endpoints: QWEN_URL and GEMMA_URL (defaults below), e.g. QWEN_URL=http://localhost:8001/v1

set -e
cd "$(dirname "$0")/.."   # project root

export HOST_UID="$(id -u)" HOST_GID="$(id -g)"
COMPOSE="docker compose -f infra/docker-compose.yml"
RUN="$COMPOSE run --rm -T"

QWEN_URL="${QWEN_URL:-http://localhost:8001/v1}"
GEMMA_URL="${GEMMA_URL:-http://localhost:8002/v1}"
QWEN_MODEL="unsloth/Qwen3.8-27B-NVFP4"
GEMMA_MODEL="nvidia/Gemma-4-26B-A4B-NVFP4"
SEEDS="${SEEDS:-11 22 33 44 55}"
INPUT=/app/datasets/instances.jsonl
OUT=/app/results/probes
WHAT="${1:-check}"

mkdir -p .hf_cache results/probes/logs   # created by you, so the container can write there

check() {   # $1 = base url, $2 = expected model id
  local body got
  body=$(curl -fsS -m 10 "$1/models") \
    || { echo "UNREACHABLE: $1 (not reachable from this machine)"; return 1; }
  got=$(echo "$body" | grep -o '"id": *"[^"]*"' | head -1)
  if [[ "$got" == *"$2"* ]]; then
    echo "ok  $1 -> $got"
  else
    echo "WRONG MODEL at $1: ${got:-no model id in the response} (expected $2)"
    echo "    response: ${body:0:300}"
    return 1
  fi
}

batch() {   # $1 = url, $2 = model id, $3 = seed, then extra cli args
  local url="$1" model="$2" seed="$3"; shift 3
  $RUN -e VLLM_BASE_URL="$url" -e TRIAGENT_SEED="$seed" triagent-runner \
    batch --input "$INPUT" --backbone "$model" --provider vllm --skip-prefetch "$@"
}

seed_pair() {   # $1 = url, $2 = model id, $3 = short tag, $4 = seed
  for abl in full no_knowledge; do
    batch "$1" "$2" "$4" --ablation "$abl" --resume \
      --output "$OUT/probeB/$3__${abl}__seed$4/results.jsonl"
  done
}

# ignore fields that differ between runs by construction (ids, timing)
strip() { sed 's/"run_id": "[^"]*"//g; s/"process_id": "[^"]*"//g; s/"latency_s": [0-9.]*//g' "$1"; }

case "$WHAT" in
  build)
    $COMPOSE build
    ;;
  check)
    rc=0
    check "$QWEN_URL" "$QWEN_MODEL" || rc=1
    check "$GEMMA_URL" "$GEMMA_MODEL" || rc=1
    exit $rc
    ;;
  smoke)
    check "$QWEN_URL" "$QWEN_MODEL"
    check "$GEMMA_URL" "$GEMMA_MODEL"
    SM="2026-10_smoke_$(date +%H%M%S)"
    for spec in "qwen38|$QWEN_URL|$QWEN_MODEL" "gemma4|$GEMMA_URL|$GEMMA_MODEL"; do
      IFS='|' read -r tag url model <<< "$spec"
      for r in a b; do
        batch "$url" "$model" 1 --ablation full --limit 2 \
          --output "/app/results/$SM/${tag}_$r/results.jsonl"
      done
      d="results/$SM/$tag"
      grep -o '"seed": [^,]*\|"served_model_start": "[^"]*"' "${d}_a/manifest.json"
      if diff <(strip "${d}_a/results.jsonl") <(strip "${d}_b/results.jsonl") > /dev/null; then
        echo "$tag: same seed -> same output"
      else
        echo "$tag: same seed -> different output"
      fi
      echo "$tag: <think> occurrences (must be 0): $(cat "${d}_a/results.jsonl" "${d}_b/results.jsonl" | grep -c '<think>' || true)"
    done
    ;;
  A)
    # monolithic baseline at T=0 ("deterministic" settings), 5 repetitions per model
    # ONLY=gemma4 or ONLY=qwen38 restricts the probe to one model
    for spec in "qwen38|$QWEN_URL|$QWEN_MODEL" "gemma4|$GEMMA_URL|$GEMMA_MODEL"; do
      IFS='|' read -r tag url model <<< "$spec"
      [[ -n "$ONLY" && "$ONLY" != "$tag" ]] && continue
      check "$url" "$model"
      for k in 1 2 3 4 5; do
        echo "[A] $tag monolithic T=0, repetition $k"
        $RUN -e VLLM_BASE_URL="$url" triagent-runner monolithic --input "$INPUT" \
          --output "$OUT/probeA/${tag}__monolithic__rep$k/results.jsonl" \
          --backbone "$model" --provider vllm --resume
      done
    done
    ;;
  B)
    check "$QWEN_URL" "$QWEN_MODEL"
    check "$GEMMA_URL" "$GEMMA_MODEL"
    # ONLY=gemma4 or ONLY=qwen38 restricts the probe to one model; SEEDS="..." picks the seeds
    for s in $SEEDS; do
      [[ -z "$ONLY" || "$ONLY" == qwen38 ]] && \
        seed_pair "$QWEN_URL"  "$QWEN_MODEL"  qwen38 "$s" > "results/probes/logs/qwen38_seed$s.log" 2>&1 &
      [[ -z "$ONLY" || "$ONLY" == gemma4 ]] && \
        seed_pair "$GEMMA_URL" "$GEMMA_MODEL" gemma4 "$s" > "results/probes/logs/gemma4_seed$s.log" 2>&1 &
    done
    wait
    ;;
  analyze)
    $RUN --entrypoint python triagent-runner experiments/analyze_probes.py "$OUT"
    ;;
  *)
    echo "usage: bash infra/run_probes.sh build|check|smoke|A|B|analyze"; exit 1
    ;;
esac
echo "done ($WHAT)"
