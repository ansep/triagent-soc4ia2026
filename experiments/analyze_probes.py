"""
Analysis of the SOC4IA probes (idea doc, Sect. 12). Standard library only.

    python3 experiments/analyze_probes.py results/probes
    python3 experiments/analyze_probes.py --selftest

Probe A (single-call baseline, T=0, repeated): how often the repeated outputs coincide.
Probe B (full vs no_knowledge, several seeds): gateway accuracy per seed,
per-instance success distribution, ICC(1) and the effective sample size it implies.
Scoring reuses evaluate.score_instance.
"""
import sys
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import score_instance, load_ground_truth, GT_PATH  # noqa: E402


INFRA_ERRORS = ("connection", "timed out", "timeout", "502", "503", "unreachable")


def is_infra_error(row: dict) -> bool:
    """Transport failure (dropped tunnel, server down): retried with --resume, never scored."""
    return any(k in (row.get("error") or "").lower() for k in INFRA_ERRORS)


def load_runs(folder: Path, keep_agent_errors: bool = False) -> dict:
    """{run_folder_name: {instance_id: row}}; checks served model.
    Infra errors are always dropped. Agent errors (e.g. a malformed tool call) are kept
    only if keep_agent_errors, so that they can be scored as failures; a successful row
    for the same instance (a --resume retry) takes precedence."""
    runs = {}
    for f in sorted(folder.glob("*/results.jsonl")):
        rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        ok = {}
        for r in rows:
            if r.get("result") and not r.get("error"):
                ok[r["instance_id"]] = r
            elif keep_agent_errors and not is_infra_error(r) and r["instance_id"] not in ok:
                ok[r["instance_id"]] = r
                print(f"  {f.parent.name}: {r['instance_id']} agent error scored as failure:"
                      f" {r['error'][:100]}")
        runs[f.parent.name] = ok
        man_file = f.parent / "manifest.json"
        if man_file.exists():
            man = json.load(open(man_file, encoding="utf-8"))
            start, end = man.get("served_model_start"), man.get("served_model_end")
            if start != end:
                print(f"WARNING {f.parent.name}: served model changed during the run: {start} -> {end}")
            elif start:
                print(f"  {f.parent.name}: served {start}")
    return runs


def icc1(groups: list[list[float]]) -> float:
    """One-way random-effects ICC(1); rows = instances, columns = repetitions (equal size)."""
    n, k = len(groups), len(groups[0])
    grand = statistics.fmean(x for g in groups for x in g)
    means = [statistics.fmean(g) for g in groups]
    msb = k * sum((m - grand) ** 2 for m in means) / (n - 1)
    msw = sum((x - m) ** 2 for g, m in zip(groups, means) for x in g) / (n * (k - 1))
    denom = msb + (k - 1) * msw
    return (msb - msw) / denom if denom else float("nan")


def probe_a(folder: Path, gt: dict) -> None:
    by_model = defaultdict(dict)  # model tag -> {run_name: {instance_id: row}}
    for name, rows in load_runs(folder).items():
        by_model[name.split("__")[0]][name] = rows
    for model, runs in sorted(by_model.items()):
        print(f"\n[{model}]")
        _probe_a_one(runs, gt)


def _probe_a_one(runs: dict, gt: dict) -> None:
    common = set.intersection(*(set(r) for r in runs.values()))
    if not common:
        print("  no instance completed in every repetition")
        return
    text = lambda n, i: (runs[n][i]["result"]["diagnosis"], runs[n][i]["result"]["treatment"])
    score = lambda n, i: tuple(sorted(score_instance(runs[n][i]["result"], gt[i]).items()))
    same_text = sum(1 for i in common if len({text(n, i) for n in runs}) == 1)
    same_score = sum(1 for i in common if len({score(n, i) for n in runs}) == 1)
    print(f"\nPROBE A: {len(runs)} repetitions at T=0, {len(common)} common instances")
    print(f"  identical output text in every repetition: {same_text}/{len(common)}")
    print(f"  identical scores in every repetition:      {same_score}/{len(common)}")
    for n in runs:
        sc = [score_instance(runs[n][i]["result"], gt[i]) for i in common]
        print(f"  {n}: diagnosis {statistics.fmean(s['diagnosis_ok'] for s in sc):.2f}"
              f"  substance {statistics.fmean(s['substance_ok'] for s in sc):.2f}"
              f"  dose {statistics.fmean(s['dose_ok'] for s in sc):.2f}")


def probe_b(folder: Path, gt: dict) -> None:
    runs = load_runs(folder, keep_agent_errors=True)
    if not runs:
        return
    by_cell = defaultdict(dict)  # (model, ablation) -> seed -> {instance_id: gateway_ok}
    for name, rows in runs.items():
        model, abl, seed = name.split("__")
        by_cell[(model, abl)][seed] = {
            i: bool(r.get("result")) and score_instance(r["result"], gt[i])["gateway_ok"]
            for i, r in rows.items()}
    print("\nPROBE B: gateway accuracy")
    for (model, abl), seeds in sorted(by_cell.items()):
        names = sorted(seeds)
        common = sorted(set.intersection(*(set(seeds[s]) for s in names)))
        m, n = len(names), len(common)
        if not n:
            print(f"  {model} {abl}: no instance completed in every seed yet")
            continue
        accs = [statistics.fmean(seeds[s][i] for i in common) for s in names]
        groups = [[float(seeds[s][i]) for s in names] for i in common]
        hist = Counter(round(statistics.fmean(g) * m) for g in groups)
        print(f"  {model} {abl} ({n} instances x {m} seeds)")
        print("    per seed: " + ", ".join(f"{s}={a:.2f}" for s, a in zip(names, accs)))
        if m > 1:
            rho = icc1(groups)
            print(f"    mean {statistics.fmean(accs):.3f}  sd {statistics.stdev(accs):.3f}"
                  f"  range {min(accs):.2f}-{max(accs):.2f}")
            print("    instances by success count: "
                  + ", ".join(f"{j}/{m}: {hist.get(j, 0)}" for j in range(m + 1)))
            cap = f"{n / rho:.0f}" if rho > 0 else "n/a"
            print(f"    ICC(1) = {rho:.2f}: {n * m} runs = {n * m / (1 + (m - 1) * rho):.0f}"
                  f" effective observations (cap with unlimited seeds: {cap})")
            print("    first k seeds: mean  sd  95% CI half-width  IQM")
            for k in range(2, m + 1):
                a = accs[:k]
                sd = statistics.stdev(a)
                print(f"      k={k:<2} {statistics.fmean(a):.3f}  {sd:.3f}"
                      f"  ±{_t95(k - 1) * sd / k ** 0.5:.3f}  {_iqm(a):.3f}")
    for model in sorted({m for m, _ in by_cell}):
        f, k = by_cell.get((model, "full")), by_cell.get((model, "no_knowledge"))
        if f and k:
            diffs = [(s, statistics.fmean(f[s].values()) - statistics.fmean(k[s].values()))
                     for s in sorted(f) if f[s] and k.get(s)]
            print(f"  {model} Full minus No-Knowledge per seed: "
                  + ", ".join(f"{s}={d:+.2f}" for s, d in diffs))


_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042}


def _t95(df: int) -> float:
    """Two-sided 95% Student t critical value (table, conservative between entries)."""
    return _T95.get(df) or next((v for d, v in sorted(_T95.items()) if d >= df), 1.96)


def _iqm(xs: list[float]) -> float:
    """Interquartile mean: drop the lowest and highest quarter, average the rest."""
    xs = sorted(xs)
    cut = len(xs) // 4
    return statistics.fmean(xs[cut:len(xs) - cut])


def _selftest() -> None:
    assert _iqm([0, 1, 1, 1, 1, 1, 1, 10]) == 1 and abs(_t95(4) - 2.776) < 1e-9 and _t95(11) == 2.179
    # all variance between instances -> ICC 1
    assert abs(icc1([[1, 1, 1], [0, 0, 0], [1, 1, 1], [0, 0, 0]]) - 1) < 1e-9
    # every instance has the same mean -> no between-instance variance -> ICC <= 0
    assert icc1([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 0, 1]]) <= 0
    assert is_infra_error({"error": "Connection error."})
    assert not is_infra_error({"error": "get_disease_info() got an unexpected keyword argument"})
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
        sys.exit()
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/probes")
    gt = load_ground_truth(GT_PATH)
    probe_a(root / "probeA", gt)
    probe_b(root / "probeB", gt)
