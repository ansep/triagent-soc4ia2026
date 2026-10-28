import os
import time
from typing import Callable, Optional
from tools.weather_api import get_weather
from tools.eppo_api import get_disease_info, send_sms

MAX_RETRIES = 2


class ExecutionAgent:

    def __init__(self):
        self.tool_registry: dict[str, Callable] = {}
        self.execution_log = []
        self._register_default_tools()

    def _register_default_tools(self):
        self.register_tool("get_weather", get_weather)
        self.register_tool("get_disease_info", get_disease_info)
        self.register_tool("send_sms", send_sms)
        self.register_tool("get_treatment_protocol", self._get_treatment_protocol)
        self.register_tool("calculate_dosage", self._calculate_dosage)
        self.register_tool("check_waiting_period", self._check_waiting_period)

    def register_tool(self, name: str, func: Callable):
        self.tool_registry[name] = func

    def execute(self, tool_name: str, tool_params: dict) -> dict:
        if tool_name not in self.tool_registry:
            return {"success": False, "error": f"unknown tool: {tool_name}", "escalate": True}

        func = self.tool_registry[tool_name]
        last_error = None

        for attempt in range(1, MAX_RETRIES + 2):
            try:
                result = func(**tool_params)
                self.execution_log.append({"tool": tool_name, "attempt": attempt, "success": True})
                return {"success": True, "tool": tool_name, "result": result, "escalate": False}
            except Exception as e:
                last_error = str(e)
                print(f"{tool_name} failed (attempt {attempt}): {e}")
                if attempt <= MAX_RETRIES:
                    time.sleep(0.5 * attempt)

        # still failing after retries: let the orchestrator know
        self.execution_log.append({"tool": tool_name, "success": False, "error": last_error})
        return {
            "success": False,
            "tool": tool_name,
            "error": last_error,
            "escalate": True,
            "message": f"{tool_name} failed after {MAX_RETRIES + 1} attempts"
        }

    def get_available_tools(self) -> list:
        return list(self.tool_registry.keys())

    # built-in agricultural helpers

    def _get_treatment_protocol(self, disease: str, severity: str = "medium") -> dict:
        protocols = {
            "wheat_brown_rust": {
                "low":    {"fungicide": "Tebuconazole 250g/L", "dose": "0.5L/ha",  "timing": "early morning"},
                "medium": {"fungicide": "Tebuconazole 250g/L", "dose": "0.75L/ha", "timing": "early morning"},
                "high":   {"fungicide": "Propiconazole 250 EC", "dose": "1.0L/ha", "timing": "immediate"}
            },
            "tomato_late_blight": {
                "low":    {"fungicide": "Mancozeb 80%",          "dose": "2.0kg/ha", "timing": "preventive"},
                "medium": {"fungicide": "Cymoxanil + Mancozeb",  "dose": "2.5kg/ha", "timing": "every 7 days"},
                "high":   {"fungicide": "Metalaxyl + Mancozeb",  "dose": "3.0kg/ha", "timing": "every 5 days"}
            },
            "olive_fruit_fly": {
                "low":    {"treatment": "Bait spray",   "dose": "1L/ha",   "timing": "weekly"},
                "medium": {"treatment": "Deltamethrin", "dose": "0.3L/ha", "timing": "bi-weekly"},
                "high":   {"treatment": "Dimethoate",   "dose": "1.5L/ha", "timing": "immediate"}
            }
        }
        key = disease.lower().replace(" ", "_")
        proto = protocols.get(key, {}).get(severity)
        if not proto:
            raise ValueError(f"no protocol for {disease} / {severity}")
        return proto

    def _calculate_dosage(self, product: str, concentration: str, area_ha: float, severity: str = "medium") -> dict:
        dose_per_ha = float(concentration.replace("L/ha", "").replace("kg/ha", ""))
        unit = "L" if "L/ha" in concentration else "kg"
        return {
            "product": product,
            "dose_per_ha": f"{dose_per_ha}{unit}/ha",
            "total_area_ha": area_ha,
            "total_required": f"{dose_per_ha * area_ha:.1f}{unit}"
        }

    def _check_waiting_period(self, product: str, harvest_date: str) -> dict:
        periods = {
            "tebuconazole": 35, "propiconazole": 28, "mancozeb": 14,
            "metalaxyl": 21,    "dimethoate": 21,    "deltamethrin": 7
        }
        days = periods.get(product.lower().split()[0], 30)
        return {
            "product": product,
            "waiting_period_days": days,
            "safe_deadline": f"{days} days before {harvest_date}"
        }
