"""
Evaluate TriAgent experiment results against ground truth.

Usage:
    # evaluate a single run folder
    python experiments/evaluate.py --results results/probes/probeB

    # evaluate all run folders under results/
    python experiments/evaluate.py --results results/

Output:
    results/aggregated/<run_folder>_summary.csv      overall accuracy per (backbone, ablation)
    results/aggregated/<run_folder>_per_process.csv  breakdown per process (P1–P5)
"""

import json
import re
import csv
import argparse
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent
GT_PATH = ROOT / "datasets" / "ground_truth.jsonl"
RESULTS_DIR = ROOT / "results"
AGG_DIR = RESULTS_DIR / "aggregated"

# sentinel returned by ExecutionAgent when ablation disables tool calls
_TOOL_DISABLED = "TOOL_DISABLED_FOR_ABLATION"


# scoring

_GATEWAY_YES = {"yes", "true", "applicable", "positive", "y", "1",
                "yes_treatment", "treat"}
_GATEWAY_NO = {"no", "false", "inapplicable", "negative", "n", "0",
               "no_treatment", "monitoring", "monitor_only",
               "disease_remains_low", "monitoring only"}


def _norm_gateway(val: str) -> str:
    v = str(val).strip().lower()
    if v in _GATEWAY_YES:
        return "yes"
    if v in _GATEWAY_NO:
        return "no"
    return v


def score_instance(result: dict, gt: dict) -> dict:
    """
    Score one result against its ground truth.
    Returns dict with 6 boolean dimensions + overall (mean).
    """
    expected = gt["expected"]
    s = {}

    # 1. diagnosis: any keyword present in the diagnosis field
    diagnosis = (result.get("diagnosis") or "").lower()
    s["diagnosis_ok"] = any(
        kw.lower() in diagnosis for kw in expected["diagnosis_keywords"]
    )

    # 2. treatment substance: any active ingredient mentioned in the treatment field
    treatment = (result.get("treatment") or "").lower()
    s["substance_ok"] = any(
        sub.lower() in treatment for sub in expected["treatment_substances"]
    )

    # 3. dosage: extract first numeric value from treatment and check range
    dose_range = expected["treatment_dose_per_ha"]
    dose_val = _extract_dose(treatment)
    if dose_val is not None:
        s["dose_ok"] = dose_range["min"] <= dose_val <= dose_range["max"]
    else:
        s["dose_ok"] = False

    # 4. required tool calls
    #    Primary signal: real observation injected by orchestrator when requires_tool fires.
    #    Fallback: action strings containing the tool name (some models emit "get_weather"
    #    literally; no_execution returns TOOL_DISABLED which is excluded).
    actions = [str(a).lower() for a in (result.get("actions_taken") or [])]
    trace = result.get("reasoning_trace") or []
    trace_tools = {
        str(step.get("action") or "").lower()
        for step in trace
        if isinstance(step, dict)
    }
    all_actions = set(actions) | trace_tools
    required = set(expected.get("required_tool_calls", []))
    if required:
        _ERR_MARKERS = ("Error", "failed", "Traceback", "got an unexpected", "ExecutionAgent")

        def _tool_ok(tool: str) -> bool:
            tool_lower = tool.lower()
            for step in trace:
                if not isinstance(step, dict):
                    continue
                action = str(step.get("action") or "").lower()
                obs = step.get("observation")
                if obs is None:
                    continue
                obs_str = str(obs).strip()
                if obs_str in ("", _TOOL_DISABLED):
                    continue
                if any(m in obs_str for m in _ERR_MARKERS):
                    continue
                if tool_lower in action:
                    return True
            # Fallback for get_weather: any valid injected observation counts
            # (model may write "Query real-time weather API" instead of "get_weather")
            if tool_lower == "get_weather":
                return any(
                    isinstance(step, dict)
                    and step.get("observation") is not None
                    and str(step.get("observation", "")).strip() not in ("", _TOOL_DISABLED)
                    and not any(m in str(step.get("observation", "")) for m in _ERR_MARKERS)
                    for step in trace
                )
            return False

        s["tools_ok"] = all(_tool_ok(t) for t in required)
    else:
        s["tools_ok"] = True

    # 5. gateway decisions: check reasoning_trace for gateway_decision fields
    trace_decisions = []
    for step in trace:
        if isinstance(step, dict) and step.get("gateway_decision"):
            trace_decisions.append(_norm_gateway(step["gateway_decision"]))

    expected_gw = expected.get("gateway_decisions", {})
    if expected_gw:
        expected_vals = [_norm_gateway(v) for v in expected_gw.values()]
        # simple check: all expected decisions appear somewhere in the trace
        s["gateway_ok"] = all(ev in trace_decisions for ev in expected_vals)
    else:
        s["gateway_ok"] = True

    # 6. escalation: matches expected
    s["escalation_ok"] = bool(result.get("escalated", False)) == bool(expected["should_escalate"])

    s["overall"] = round(sum(s.values()) / len(s), 4)
    return s


def _extract_dose(text: str) -> float | None:
    """Extract first decimal number from a string (e.g. '0.75 L/ha' -> 0.75)."""
    m = re.search(r"(\d+\.?\d*)", text)
    return float(m.group(1)) if m else None


# aggregation

SCORE_FIELDS = ["diagnosis_ok", "substance_ok", "dose_ok", "tools_ok", "gateway_ok",
                "escalation_ok", "overall"]


def load_ground_truth(path: Path = GT_PATH) -> dict[str, dict]:
    gt = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            gt[row["instance_id"]] = row
    return gt


def evaluate_run_folder(run_folder: Path, gt: dict,
                         gt_by_backbone: dict[str, dict] | None = None) -> list[dict]:
    """Walk a run folder and score every results.jsonl found.

    gt_by_backbone, if given, overrides `gt` on a per-backbone basis: used by the
    tool-forcing dataset, where the weather-dependent gateway decisions depend on what
    each backbone's run actually observed from the live get_weather call (see
    experiments/derive_toolforcing_gateway_gt.py) and therefore cannot share one GT
    file across backbones the way the production dataset does.
    """
    rows = []
    for results_file in run_folder.rglob("results.jsonl"):
        manifest_file = results_file.parent / "manifest.json"
        manifest = {}
        if manifest_file.exists():
            with open(manifest_file, encoding="utf-8") as f:
                manifest = json.load(f)

        backbone = manifest.get("backbone") or results_file.parent.name.split("__")[0]
        ablation = manifest.get("ablation") or (
            results_file.parent.name.split("__")[1]
            if "__" in results_file.parent.name else "full"
        )
        cell_gt = (gt_by_backbone or {}).get(backbone, gt)

        with open(results_file, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("error") or not row.get("result"):
                    continue
                iid = row["instance_id"]
                if iid not in cell_gt:
                    continue
                scores = score_instance(row["result"], cell_gt[iid])
                rows.append({
                    "backbone": backbone,
                    "ablation": ablation,
                    "process_id": row.get("process_id", iid[:2]),
                    "instance_id": iid,
                    "latency_s": row.get("latency_s", 0),
                    "cost_usd": row["result"].get("cost_usd", 0),
                    "escalated": row["result"].get("escalated", False),
                    **scores,
                })
    return rows


def aggregate(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Compute per-(backbone, ablation) and per-(backbone, ablation, process_id) aggregates."""
    # group
    by_cell = defaultdict(list)
    by_process = defaultdict(list)
    for r in rows:
        key = (r["backbone"], r["ablation"])
        by_cell[key].append(r)
        by_process[(r["backbone"], r["ablation"], r["process_id"])].append(r)

    def _mean(lst, field):
        vals = [r[field] for r in lst]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    summary = []
    for (bb, abl), group in sorted(by_cell.items()):
        summary.append({
            "backbone": bb,
            "ablation": abl,
            "n": len(group),
            **{f: _mean(group, f) for f in SCORE_FIELDS},
            "mean_latency_s": _mean(group, "latency_s"),
            "mean_cost_usd": _mean(group, "cost_usd"),
            "escalation_rate": round(sum(r["escalated"] for r in group) / len(group), 4),
        })

    per_proc = []
    for (bb, abl, pid), group in sorted(by_process.items()):
        per_proc.append({
            "backbone": bb,
            "ablation": abl,
            "process_id": pid,
            "n": len(group),
            **{f: _mean(group, f) for f in SCORE_FIELDS},
            "mean_latency_s": _mean(group, "latency_s"),
            "escalation_rate": round(sum(r["escalated"] for r in group) / len(group), 4),
        })

    return summary, per_proc


def write_csv(rows: list[dict], path: Path):
    if not rows:
        print(f"no rows to write to {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"written {len(rows)} rows -> {path}")


# CLI

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True,
                        help="Path to a run folder or to results/ (all subfolders scanned)")
    parser.add_argument("--gt", default=None,
                        help="Path to ground truth JSONL (default: datasets/ground_truth.jsonl)")
    args = parser.parse_args()

    gt_path = Path(args.gt) if args.gt else GT_PATH
    gt = load_ground_truth(gt_path)
    print(f"loaded {len(gt)} ground truth entries from {gt_path.name}")

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"results path not found: {results_path}")
        return

    # collect run folders
    if (results_path / "results.jsonl").exists() or list(results_path.rglob("results.jsonl")):
        run_folders = [results_path]
    else:
        run_folders = [d for d in results_path.iterdir() if d.is_dir()]

    all_rows = []
    for folder in run_folders:
        rows = evaluate_run_folder(folder, gt)
        print(f"{folder.name}: {len(rows)} scored instances")
        all_rows.extend(rows)

    if not all_rows:
        print("no results found")
        return

    summary, per_proc = aggregate(all_rows)

    run_tag = results_path.name if results_path.is_dir() else results_path.parent.name
    write_csv(summary, AGG_DIR / f"{run_tag}_summary.csv")
    write_csv(per_proc, AGG_DIR / f"{run_tag}_per_process.csv")

    # print quick table
    print("\n-- SUMMARY ------------------------------------------------------------------")
    print(f"{'backbone':<30} {'ablation':<15} {'n':>5} {'overall':>8} {'diagnosis':>10} {'substance':>10} {'dose':>6} {'gateway':>8}")
    for r in summary:
        print(f"{r['backbone']:<30} {r['ablation']:<15} {r['n']:>5} "
              f"{r['overall']:>8.3f} {r['diagnosis_ok']:>10.3f} {r['substance_ok']:>10.3f} "
              f"{r['dose_ok']:>6.3f} {r['gateway_ok']:>8.3f}")


if __name__ == "__main__":
    main()
