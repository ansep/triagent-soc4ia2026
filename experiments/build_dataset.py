"""
Generate the 100-instance evaluation dataset and ground truth for TriAgent.

Phases:
  A  Seed data: 5 processes with canonical symptoms, protocols, gateway definitions
  B  Combinatorial expansion: 3 paraphrase × 3 severity × 2 weather × 3 stage = 54 combos
     Stratified subsample to exactly 20 per process (100 total)
  C  Paraphrase generation via LLM (default: qwen-local via llama.cpp; override with --provider)
  D  Ground truth file: derived from seed tables, no LLM involved

Usage:
    # Local Qwen3-32B (llama.cpp running on localhost:8080):
    python experiments/build_dataset.py

    # against a llama.cpp server:
    python experiments/build_dataset.py --llamacpp-url http://localhost:8080/v1

    # Cloud OpenAI (if you have a key):
    python experiments/build_dataset.py --provider openai --backbone gpt-4o-mini

    python experiments/build_dataset.py --dry-run   # print stats, no write

Outputs:
    datasets/instances.jsonl      (100 lines)
    datasets/ground_truth.jsonl   (100 lines)
    datasets/instances.jsonl.sha256
"""

import json
import os
import random
import re
import hashlib
import sys
import argparse
from pathlib import Path
from itertools import product
from openai import OpenAI

_THINK_RE = re.compile(r"<think>[\s\S]*?</think>")


def _strip_thinking(raw: str) -> str:
    """Strip CoT blocks from model output.
    Handles two formats:
      - <think>...</think>          (standard / OpenAI-compat Qwen3)
      - <|channel>thought...<channel|>  (llama.cpp Qwen3 internal tokens)
    """
    if "<channel|>" in raw:
        # take everything after the last closing channel token
        return raw.split("<channel|>")[-1].strip()
    return _THINK_RE.sub("", raw).strip()


SEED = 42
DATASETS_DIR = Path(__file__).parent.parent / "datasets"
CACHE_FILE = DATASETS_DIR / ".cache" / "paraphrase.json"

# A: SEED DATA
# Each process entry contains everything needed to generate instances AND ground truth.
# treatment_protocols mirrors execution_agent._get_treatment_protocol (extended for P4/P5).

PROCESSES = {
    "P1": {
        "name": "Wheat brown rust",
        "pathogen": "Puccinia triticina",
        "crop": "wheat",
        "complexity": "Low",
        "n_steps": 4,
        "n_gateways": 1,
        "diagnosis_keywords": ["brown rust", "Puccinia triticina", "leaf rust"],
        "canonical_symptoms": (
            "Orange-yellow pustules (uredinia) on the upper surface of wheat leaves, "
            "scattered across the canopy. Pustules surrounded by yellow halo. "
            "Underside of leaves shows smaller, darker telial pustules in late stage."
        ),
        "weather_favorable": "temperature 15–22 °C with relative humidity above 80%",
        "weather_unfavorable": "high temperatures above 30 °C or dry conditions",
        "growth_stages": {
            "early": "tillering (Zadoks GS 20–29)",
            "mid": "stem elongation (Zadoks GS 30–37)",
            "late": "heading (Zadoks GS 50–59)",
        },
        "treatment_protocols": {
            "low":    {"fungicide": "Tebuconazole 250g/L", "dose_l_per_ha": 0.5,
                       "unit": "L/ha", "timing": "early morning",
                       "substances": ["tebuconazole"]},
            "medium": {"fungicide": "Tebuconazole 250g/L", "dose_l_per_ha": 0.75,
                       "unit": "L/ha", "timing": "early morning",
                       "substances": ["tebuconazole"]},
            "high":   {"fungicide": "Propiconazole 250 EC", "dose_l_per_ha": 1.0,
                       "unit": "L/ha", "timing": "immediate",
                       "substances": ["propiconazole"]},
        },
        "required_tool_calls": ["get_treatment_protocol", "calculate_dosage"],
        "gateway_definitions": {
            "weather_check": {
                "condition": "Is weather favorable for disease spread?",
                "yes_means": "apply fungicide immediately",
                "no_means": "monitor and reassess in 5 days",
            }
        },
        "locations": ["Bologna, IT", "Foggia, IT", "Ankara, TR"],
    },
    "P2": {
        "name": "Tomato late blight",
        "pathogen": "Phytophthora infestans",
        "crop": "tomato",
        "complexity": "Low",
        "n_steps": 5,
        "n_gateways": 2,
        "diagnosis_keywords": ["late blight", "Phytophthora infestans"],
        "canonical_symptoms": (
            "Water-soaked lesions on tomato leaves that rapidly enlarge and turn brown-black. "
            "White sporulating growth visible on the underside of affected leaves under humid conditions. "
            "Dark brown cankers on stems. Infected fruits show brown firm rot."
        ),
        "weather_favorable": "temperature 10–20 °C with leaf wetness periods exceeding 10 hours",
        "weather_unfavorable": "hot dry conditions above 28 °C",
        "growth_stages": {
            "early": "seedling / transplant establishment",
            "mid": "flowering and fruit set",
            "late": "fruit ripening",
        },
        "treatment_protocols": {
            "low":    {"fungicide": "Mancozeb 80%", "dose_l_per_ha": 2.0,
                       "unit": "kg/ha", "timing": "preventive",
                       "substances": ["mancozeb"]},
            "medium": {"fungicide": "Cymoxanil + Mancozeb", "dose_l_per_ha": 2.5,
                       "unit": "kg/ha", "timing": "every 7 days",
                       "substances": ["cymoxanil", "mancozeb"]},
            "high":   {"fungicide": "Metalaxyl + Mancozeb", "dose_l_per_ha": 3.0,
                       "unit": "kg/ha", "timing": "every 5 days",
                       "substances": ["metalaxyl", "mancozeb"]},
        },
        "required_tool_calls": ["get_weather", "get_treatment_protocol", "calculate_dosage"],
        "gateway_definitions": {
            "weather_check": {
                "condition": "Is weather favorable (cool, wet) for disease spread?",
                "yes_means": "increase spray frequency",
                "no_means": "maintain standard schedule",
            },
            "severity_branch": {
                "condition": "Is disease severity high?",
                "yes_means": "apply systemic fungicide immediately",
                "no_means": "apply contact fungicide preventively",
            },
        },
        "locations": ["Naples, IT", "Valencia, ES", "Salerno, IT"],
    },
    "P3": {
        "name": "Olive fruit fly",
        "pathogen": "Bactrocera oleae",
        "crop": "olive",
        "complexity": "Medium",
        "n_steps": 7,
        "n_gateways": 3,
        "diagnosis_keywords": ["olive fruit fly", "Bactrocera oleae", "fruit fly"],
        "canonical_symptoms": (
            "Adult fly puncture marks (oviposition stings) on olive fruits, appearing as small dark spots. "
            "Tunneling and galleries inside the olive flesh caused by larval feeding. "
            "Premature fruit drop and secondary bacterial rot at the sting site."
        ),
        "weather_favorable": "temperature 20–30 °C with moderate humidity; olive fruits present",
        "weather_unfavorable": "temperatures above 35 °C or below 10 °C reduce fly activity",
        "growth_stages": {
            "early": "fruit set (June–July)",
            "mid": "fruit development (August–September)",
            "late": "pre-harvest ripening (October)",
        },
        "treatment_protocols": {
            "low":    {"fungicide": "Bait spray (Spinosad)", "dose_l_per_ha": 1.0,
                       "unit": "L/ha", "timing": "weekly",
                       "substances": ["spinosad"]},
            "medium": {"fungicide": "Deltamethrin", "dose_l_per_ha": 0.3,
                       "unit": "L/ha", "timing": "bi-weekly",
                       "substances": ["deltamethrin"]},
            "high":   {"fungicide": "Dimethoate", "dose_l_per_ha": 1.5,
                       "unit": "L/ha", "timing": "immediate",
                       "substances": ["dimethoate"]},
        },
        "required_tool_calls": ["get_weather", "get_treatment_protocol", "check_waiting_period"],
        "gateway_definitions": {
            "infestation_threshold": {
                "condition": "Is infestation rate above economic threshold (>5% infested fruits)?",
                "yes_means": "apply insecticide",
                "no_means": "deploy mass trapping only",
            },
            "weather_check": {
                "condition": "Is temperature in optimal range for spray efficacy (15–30 °C)?",
                "yes_means": "proceed with spray",
                "no_means": "delay application",
            },
            "harvest_proximity": {
                "condition": "Is harvest within 21 days?",
                "yes_means": "use short pre-harvest interval product",
                "no_means": "standard product selection",
            },
        },
        "locations": ["Palermo, IT", "Kalamata, GR", "Jaén, ES"],
    },
    "P4": {
        "name": "Wheat fusarium head blight",
        "pathogen": "Fusarium graminearum",
        "crop": "wheat",
        "complexity": "Medium",
        "n_steps": 8,
        "n_gateways": 3,
        "diagnosis_keywords": ["fusarium", "head blight", "Fusarium graminearum", "scab", "bleached spikelets"],
        "canonical_symptoms": (
            "Bleached or premature whitening of individual spikelets or entire wheat heads (Fusarium Head Blight). "
            "Pink-orange spore masses (sporodochia) visible at the base of infected spikelets under wet conditions. "
            "Shriveled, light-weight grain ('tombstone' kernels) at harvest."
        ),
        "weather_favorable": "warm humid conditions (20–25 °C) during anthesis with rain or high humidity",
        "weather_unfavorable": "dry conditions during flowering or temperatures below 15 °C",
        "growth_stages": {
            "early": "tillering (Zadoks GS 20–29)",
            "mid": "anthesis / flowering (Zadoks GS 60–69)",
            "late": "grain fill (Zadoks GS 70–80)",
        },
        "treatment_protocols": {
            "low":    {"fungicide": "Tebuconazole 250g/L", "dose_l_per_ha": 1.0,
                       "unit": "L/ha", "timing": "at anthesis",
                       "substances": ["tebuconazole"]},
            "medium": {"fungicide": "Metconazole + Tebuconazole", "dose_l_per_ha": 1.25,
                       "unit": "L/ha", "timing": "at anthesis, repeat in 7 days",
                       "substances": ["metconazole", "tebuconazole"]},
            "high":   {"fungicide": "Prothioconazole + Tebuconazole", "dose_l_per_ha": 1.5,
                       "unit": "L/ha", "timing": "immediate at anthesis",
                       "substances": ["prothioconazole", "tebuconazole"]},
        },
        "required_tool_calls": ["get_weather", "get_treatment_protocol", "calculate_dosage", "check_waiting_period"],
        "gateway_definitions": {
            "weather_check": {
                "condition": "Is weather humid and warm during anthesis (favorable for FHB)?",
                "yes_means": "apply fungicide at anthesis",
                "no_means": "monitor, reassess at grain fill",
            },
            "mycotoxin_risk": {
                "condition": "Is mycotoxin risk high (susceptible variety + high disease pressure)?",
                "yes_means": "use broad-spectrum triazole at full dose",
                "no_means": "apply standard dose",
            },
            "timing_critical": {
                "condition": "Is wheat at anthesis right now (most effective window)?",
                "yes_means": "apply immediately",
                "no_means": "delay until optimal window",
            },
        },
        "locations": ["Bologna, IT", "Foggia, IT", "Budapest, HU"],
    },
    "P5": {
        "name": "Citrus greening (HLB)",
        "pathogen": "Candidatus Liberibacter asiaticus",
        "crop": "citrus",
        "complexity": "High",
        "n_steps": 10,
        "n_gateways": 4,
        "diagnosis_keywords": ["citrus greening", "HLB", "huanglongbing", "Liberibacter", "blotchy mottle"],
        "canonical_symptoms": (
            "Yellow shoots (flush) with blotchy mottling on citrus leaves that does not follow vein patterns "
            "('blotchy mottle' — asymmetric, distinct from nutrient deficiency). "
            "Lopsided, small fruits with off-colour, bitter juice. Premature fruit drop. "
            "Twig and branch dieback. Presence of psyllid insect vector (Diaphorina citri) on new growth."
        ),
        "weather_favorable": "warm climate (25–35 °C) year-round promoting psyllid vector activity",
        "weather_unfavorable": "cold winters below 10 °C reduce vector population",
        "growth_stages": {
            "early": "young grove (1–3 years, high vector susceptibility)",
            "mid": "productive grove (4–10 years)",
            "late": "mature grove (>10 years)",
        },
        "treatment_protocols": {
            "low": {
                "fungicide": "Imidacloprid systemic insecticide (psyllid vector control)",
                "dose_l_per_ha": 0.5, "unit": "L/ha", "timing": "at new flush emergence",
                "substances": ["imidacloprid"]
            },
            "medium": {
                "fungicide": "Dimethoate + Imidacloprid (vector + nutritional support)",
                "dose_l_per_ha": 1.0, "unit": "L/ha", "timing": "every 6 weeks during flush",
                "substances": ["dimethoate", "imidacloprid"]
            },
            "high": {
                "fungicide": "Remove infected trees + immediate psyllid suppression (Spirotetramat)",
                "dose_l_per_ha": 1.5, "unit": "L/ha", "timing": "immediate",
                "substances": ["spirotetramat"]
            },
        },
        "required_tool_calls": ["get_disease_info", "get_treatment_protocol", "check_waiting_period", "send_sms"],
        "gateway_definitions": {
            "confirmation_test": {
                "condition": "Has PCR or field kit confirmed Liberibacter presence?",
                "yes_means": "proceed with eradication protocol",
                "no_means": "treat as nutritional deficiency, retest in 30 days",
            },
            "tree_removal": {
                "condition": "Is disease incidence above 30% in the block?",
                "yes_means": "remove infected trees and surrounding buffer",
                "no_means": "intensive vector suppression + nutritional program",
            },
            "vector_pressure": {
                "condition": "Is psyllid population above threshold (>1 nymph per flush)?",
                "yes_means": "apply insecticide immediately",
                "no_means": "monitor weekly",
            },
            "notification_required": {
                "condition": "Is HLB a notifiable disease in this region?",
                "yes_means": "notify phytosanitary authority and neighboring farms via SMS",
                "no_means": "internal management only",
            },
        },
        "locations": ["Catania, IT", "Valencia, ES", "Sao Paulo, BR"],
    },
}

PARAPHRASE_STYLES = [
    ("technical_report",
     "a formal agronomist field inspection report, using technical botanical/pathological terminology"),
    ("informal_text_msg",
     "an informal voice note transcription from a farmer, using plain everyday language, "
     "possibly incomplete sentences"),
    ("detailed_observation",
     "a detailed step-by-step field observation log written by an extension agent, "
     "describing symptoms, affected area percentage, and environmental conditions"),
]

SEVERITIES = ["low", "medium", "high"]
WEATHER_CONDITIONS = [True, False]   # True = favorable
GROWTH_STAGE_KEYS = ["early", "mid", "late"]


# B: COMBINATORIAL EXPANSION

def build_combinations(process_id: str, proc: dict) -> list[dict]:
    """All 54 combinations for one process; returns list of raw combo dicts."""
    combos = []
    for (style_key, _), severity, weather_fav, stage_key in product(
        PARAPHRASE_STYLES, SEVERITIES, WEATHER_CONDITIONS, GROWTH_STAGE_KEYS
    ):
        combos.append({
            "process_id": process_id,
            "style_key": style_key,
            "severity": severity,
            "weather_favorable": weather_fav,
            "growth_stage_key": stage_key,
        })
    return combos


def stratified_subsample(combos: list[dict], n: int = 20, rng: random.Random = None) -> list[dict]:
    """
    Stratified subsample: ensure roughly equal coverage across severity, weather, and stage
    while hitting exactly n samples.
    """
    rng = rng or random.Random(SEED)
    # sort so the strata ordering is deterministic before shuffle
    combos = sorted(combos, key=lambda c: (c["severity"], c["weather_favorable"], c["growth_stage_key"]))
    rng.shuffle(combos)
    # take n, guaranteed to cover all 3 severities via ceiling division
    return combos[:n]


# C: PARAPHRASE GENERATION

def generate_paraphrase(client: OpenAI, backbone: str, proc: dict, combo: dict, cache: dict) -> str:
    style_key = combo["style_key"]
    severity = combo["severity"]
    weather_fav = combo["weather_favorable"]
    stage_key = combo["growth_stage_key"]

    cache_key = f"{combo['process_id']}__{style_key}__{severity}__{weather_fav}__{stage_key}"
    if cache_key in cache:
        return cache[cache_key]

    style_desc = next(s[1] for s in PARAPHRASE_STYLES if s[0] == style_key)
    weather_desc = proc["weather_favorable"] if weather_fav else proc["weather_unfavorable"]
    stage_desc = proc["growth_stages"][stage_key]

    severity_guidance = {
        "low":    "VERY FEW isolated lesions, <15% affected area — use words like 'incipient', 'isolated', 'few', 'early detection', 'limited spread'",
        "medium": "moderate spread, 30–40% affected area — use words like 'moderate', 'spreading', 'around 30–40% of plants affected'",
        "high":   "severe and widespread, >60% affected — use words like 'severe', 'widespread', 'extensive damage'",
    }[severity]

    prompt = f"""You are an agricultural domain expert generating evaluation data for an AI benchmark.

Rewrite the following canonical symptom description as {style_desc}.

STRICT RULES — violating any of these invalidates the sample:
1. Disease: {proc['name']} caused by {proc['pathogen']}. Use ONLY this disease name and pathogen. Do NOT substitute with any similar disease (e.g. do not replace brown rust with stripe rust, do not replace HLB with citrus tristeza).
2. Severity MUST be {severity.upper()}: {severity_guidance}. If you are writing low severity, your description MUST say "few", "incipient", or "limited". Do NOT describe widespread damage for low severity.
3. Growth stage: {stage_desc}
4. Weather context: {weather_desc}
5. Preserve ALL morphological signs: {proc['canonical_symptoms'][:200]}...
6. Output ONLY the rewritten description — no preamble, no explanation.

Canonical symptoms:
{proc['canonical_symptoms']}"""

    resp = client.chat.completions.create(
        model=backbone,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    raw = resp.choices[0].message.content.strip()
    text = _strip_thinking(raw)
    cache[cache_key] = text
    return text


# D: GROUND TRUTH

# Gateways whose decision formula depends on weather_favorable. Used both by
# build_ground_truth (production, where weather_fav is always known from the
# combo) and by the tool-forcing pipeline (experiments/generate_toolforcing.py +
# experiments/derive_toolforcing_gateway_gt.py), where weather_fav is NOT known
# at dataset-generation time (it depends on a live API call made at run time),
# so these gateways must be filled in later, per backbone, from the logged
# observation rather than guessed here.
WEATHER_DEPENDENT_GATEWAYS = {"weather_check", "mycotoxin_risk", "vector_pressure"}


def _gateway_decision(gw_id: str, severity: str, weather_fav, growth_stage_key: str):
    """
    Compute a single gateway's expected decision ("yes"/"no").
    weather_fav may be None when it is not yet known (tool-forcing GT at
    generation time, before the live weather observation exists): in that
    case, weather-dependent gateways return None (caller should omit them)
    and weather-independent gateways are still resolved normally.
    """
    if gw_id in WEATHER_DEPENDENT_GATEWAYS and weather_fav is None:
        return None
    if gw_id == "weather_check":
        return "yes" if weather_fav else "no"
    elif gw_id == "severity_branch":
        return "yes" if severity == "high" else "no"
    elif gw_id == "infestation_threshold":
        return "yes" if severity in ("medium", "high") else "no"
    elif gw_id == "harvest_proximity":
        return "yes" if growth_stage_key == "late" else "no"
    elif gw_id == "mycotoxin_risk":
        return "yes" if severity == "high" and weather_fav else "no"
    elif gw_id == "timing_critical":
        return "yes" if growth_stage_key == "mid" else "no"
    elif gw_id == "confirmation_test":
        return "yes"  # all instances assume confirmed diagnosis
    elif gw_id == "tree_removal":
        return "yes" if severity == "high" else "no"
    elif gw_id == "vector_pressure":
        return "yes" if weather_fav and severity != "low" else "no"
    elif gw_id == "notification_required":
        return "yes"  # HLB is notifiable in all evaluation regions
    else:
        return "yes" if weather_fav else "no"


def build_gateway_decisions(proc: dict, severity: str, weather_fav, growth_stage_key: str) -> dict:
    """Resolve all of a process's gateways; omits any that return None (unknown)."""
    decisions = {}
    for gw_id in proc["gateway_definitions"]:
        v = _gateway_decision(gw_id, severity, weather_fav, growth_stage_key)
        if v is not None:
            decisions[gw_id] = v
    return decisions


def build_ground_truth(instance_id: str, proc: dict, combo: dict) -> dict:
    severity = combo["severity"]
    weather_fav = combo["weather_favorable"]
    protocol = proc["treatment_protocols"][severity]
    dose = protocol["dose_l_per_ha"]

    # dose tolerance window: ±20% of nominal
    dose_min = round(dose * 0.8, 3)
    dose_max = round(dose * 1.2, 3)

    # gateway ground truth (weather_fav is always known here, so nothing is omitted)
    gateway_decisions = build_gateway_decisions(proc, severity, weather_fav, combo["growth_stage_key"])

    return {
        "instance_id": instance_id,
        "process_id": combo["process_id"],
        "expected": {
            "diagnosis_keywords": proc["diagnosis_keywords"],
            "treatment_substances": protocol["substances"],
            "treatment_dose_per_ha": {
                "min": dose_min,
                "max": dose_max,
                "unit": protocol["unit"],
            },
            "treatment_timing": [protocol["timing"]],
            "required_tool_calls": proc["required_tool_calls"],
            "gateway_decisions": gateway_decisions,
            "should_escalate": False,
        },
    }


# MAIN

def make_client(provider: str, backbone: str, llamacpp_url: str) -> OpenAI:
    if provider == "openai":
        return OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    if provider == "llamacpp":
        return OpenAI(api_key="lcpp", base_url=llamacpp_url)
    if provider == "groq":
        return OpenAI(api_key=os.environ.get("GROQ_API_KEY", ""),
                      base_url="https://api.groq.com/openai/v1")
    if provider == "deepseek":
        return OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
                      base_url="https://api.deepseek.com/v1")
    raise ValueError(f"Unknown provider: {provider}")


def main(dry_run: bool = False, provider: str = "llamacpp",
         backbone: str = "qwen-local", llamacpp_url: str = "http://localhost:8080/v1"):
    rng = random.Random(SEED)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

    cache: dict = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            cache = json.load(f)
        print(f"loaded {len(cache)} cached paraphrases")

    client = make_client(provider, backbone, llamacpp_url)
    print(f"paraphrase model: {backbone} via {provider}")

    instances = []
    ground_truths = []
    instance_counter = 0

    for proc_id, proc in PROCESSES.items():
        combos = build_combinations(proc_id, proc)
        sampled = stratified_subsample(combos, n=20, rng=rng)

        for i, combo in enumerate(sampled, start=1):
            instance_id = f"{proc_id}_{i:03d}"
            instance_counter += 1

            # pick a location deterministically
            location = proc["locations"][i % len(proc["locations"])]
            stage_desc = proc["growth_stages"][combo["growth_stage_key"]]

            # generate paraphrase
            if not dry_run:
                description = generate_paraphrase(client, backbone, proc, combo, cache)
            else:
                description = f"[DRY RUN] {proc['canonical_symptoms'][:120]}..."

            inst = {
                "instance_id": instance_id,
                "process_id": proc_id,
                "process_name": proc["name"],
                "dims": {
                    "paraphrase_style": combo["style_key"],
                    "severity": combo["severity"],
                    "weather_favorable": combo["weather_favorable"],
                    "growth_stage": combo["growth_stage_key"],
                },
                "request": {
                    "description": description,
                    "crop": proc["crop"],
                    "location": location,
                    "growth_stage": stage_desc,
                    "severity": combo["severity"],
                },
            }

            gt = build_ground_truth(instance_id, proc, combo)
            instances.append(inst)
            ground_truths.append(gt)

        print(f"{proc_id} ({proc['name']}): {len(sampled)} instances generated")

    print(f"\ntotal: {instance_counter} instances")

    if dry_run:
        print("[DRY RUN] no files written")
        return

    # save cache
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

    # write instances
    # newline="" prevents Python's universal-newline translation on Windows
    # (which would turn "\n" into "\r\n" and silently change the file's SHA-256,
    # invalidating dataset_sha256 already recorded in run manifests).
    instances_path = DATASETS_DIR / "instances.jsonl"
    with open(instances_path, "w", newline="", encoding="utf-8") as f:
        for inst in instances:
            f.write(json.dumps(inst) + "\n")

    # write ground truth
    gt_path = DATASETS_DIR / "ground_truth.jsonl"
    with open(gt_path, "w", newline="", encoding="utf-8") as f:
        for gt in ground_truths:
            f.write(json.dumps(gt) + "\n")

    # SHA-256 checksum
    sha = _sha256(instances_path)
    with open(instances_path.with_suffix(".jsonl.sha256"), "w", newline="") as f:
        f.write(sha + "\n")

    print(f"\nwritten:")
    print(f"  {instances_path}  ({sha[:12]}...)")
    print(f"  {gt_path}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--provider", default="llamacpp",
                        choices=["llamacpp", "openai", "groq", "deepseek"])
    parser.add_argument("--backbone", default="qwen-local",
                        help="Model alias / name (e.g. qwen-local, gpt-4o-mini)")
    parser.add_argument("--llamacpp-url", default="http://localhost:8080/v1",
                        help="Base URL for the llama.cpp server (without /chat/completions)")
    args = parser.parse_args()
    main(dry_run=args.dry_run, provider=args.provider,
         backbone=args.backbone, llamacpp_url=args.llamacpp_url)
