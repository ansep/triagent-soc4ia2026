import os
import re
import json
from models.schemas import ReasoningStep
from config.llm_client import make_client
from config.settings import Provider

CONFIDENCE_THRESHOLD = 0.6
DEFAULT_MODEL = "llama-3.3-70b-versatile"

# the agent always responds in JSON: easier to parse and less hallucination-prone
SYSTEM_PROMPT = """You are an expert agricultural process reasoning agent.
You receive a structured process rule and must follow it step by step.

At each step:
1. THINK about the current state and what comes next
2. ACT by deciding what tool or action is needed
3. OBSERVE the result before moving to the next step

For each <<decision>> gateway, pick the right branch and rate your confidence (0 to 1).
If confidence < {threshold}, flag for human review instead of guessing.

TOOL CALLING RULES — READ CAREFULLY:
- If a step requires external data (e.g. weather, sensor readings), you MUST set "requires_tool": true.
- Setting "requires_tool": true causes the system to PAUSE, execute the tool in the real world, and inject the actual result into "observation" in your next step.
- NEVER fabricate or assume tool results. If you write "tool returned None" or invent values without having set "requires_tool": true in the previous step, you are hallucinating.
- Available tools: "get_weather" (params: {{"location": "<city, country>"}}).
- If no weather data is present in the context and the rule requires it, set requires_tool=true with tool_name="get_weather".

Always reply with this exact JSON:
{{
  "step": <int>,
  "thought": "<reasoning>",
  "action": "<what to do or null>",
  "gateway_decision": "<yes/no/null>",
  "confidence": <0.0-1.0>,
  "process_complete": <bool>,
  "requires_tool": <bool>,
  "tool_name": "<name or null>",
  "tool_params": {{}},
  "diagnosis": "<final diagnosis or null>",
  "treatment": "<recommendation or null>"
}}"""


def _llamacpp_extra(provider: str, model: str = "") -> dict:
    """Sampling params + cache hint for llama.cpp providers.
    Qwen3 non-thinking instruct: temperature=0.7, top_p=0.8, top_k=20.
    Gemma4 instruct: temperature=1.0, top_p=0.95 (no thinking suppression needed).
    Cloud providers: temperature=0 for reproducibility.
    TRIAGENT_SEED (env) fixes the sampling seed; unset = random seed per request.
    vLLM gets the same sampling params, with thinking off as in llama.cpp --reasoning off.
    """
    if provider and (provider.startswith("llamacpp") or provider == "vllm"):
        if "gemma" in model.lower():
            params = {
                "temperature": 1.0,
                "top_p": 0.95,
                "extra_body": {"top_k": 64},
            }
        else:
            # Qwen3 non-thinking instruct defaults
            params = {
                "temperature": 0.7,
                "top_p": 0.8,
                "extra_body": {
                    "top_k": 20,
                    "presence_penalty": 1.5,
                    "cache_prompt": True,
                },
            }
        seed = os.environ.get("TRIAGENT_SEED")
        if seed:
            params["extra_body"]["seed"] = int(seed)
        if provider == "vllm":
            params["extra_body"]["chat_template_kwargs"] = {"enable_thinking": False}
            # a step is a short JSON object; without a cap a degenerate generation holds a
            # vLLM slot until the client times out
            params["max_tokens"] = 2048
        return params
    return {"temperature": 0.0}


def _strip_thinking(raw: str) -> str:
    if "<channel|>" in raw:
        return raw.split("<channel|>")[-1].strip()
    return re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()


def _parse_json_response(raw: str, provider: str) -> dict:
    cleaned = _strip_thinking(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r'\{[\s\S]*\}', cleaned)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        raise


class ReasoningAgent:

    def __init__(self, model: str = DEFAULT_MODEL, provider: Provider = None):
        self.model = model
        self.provider = provider or "openai"
        self.client = make_client(provider)
        self.total_tokens = 0
        self.confidence_threshold = CONFIDENCE_THRESHOLD

    def run(self, process_rule: str, input_description: str, context: dict = {}) -> dict:
        reasoning_trace = []
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(threshold=self.confidence_threshold)},
            {"role": "user", "content": f"Process Rule:\n{process_rule}\n\nSituation:\n{input_description}\n\nContext:\n{json.dumps(context, indent=2)}\n\nStart executing the rule."}
        ]
        diagnosis = None
        treatment = None
        escalated = False
        actions_taken = []

        _extra = _llamacpp_extra(self.provider, self.model)
        for step_num in range(1, 20):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                **_extra
            )
            self.total_tokens += resp.usage.total_tokens
            raw = resp.choices[0].message.content

            try:
                step_data = _parse_json_response(raw, self.provider)
            except json.JSONDecodeError:
                break

            step = ReasoningStep(
                step=step_num,
                thought=step_data.get("thought", ""),
                action=step_data.get("action"),
                observation=None,
                confidence=step_data.get("confidence", 1.0),
                gateway_decision=step_data.get("gateway_decision")
            )

            # bail out if not confident enough
            if step_data.get("confidence", 1.0) < self.confidence_threshold:
                escalated = True
                step.observation = "low confidence — escalated to human"
                reasoning_trace.append(step)
                break

            if step_data.get("diagnosis"):
                diagnosis = step_data["diagnosis"]
            if step_data.get("treatment"):
                treatment = step_data["treatment"]
            if step_data.get("action"):
                actions_taken.append(step_data["action"])

            reasoning_trace.append(step)

            if step_data.get("process_complete"):
                break

            # hand off to execution agent if a tool is needed
            if step_data.get("requires_tool"):
                return {
                    "status": "tool_required",
                    "tool_name": step_data.get("tool_name"),
                    "tool_params": step_data.get("tool_params", {}),
                    "reasoning_trace": reasoning_trace,
                    "process_state": {},
                    "messages": messages,
                    "step_data": step_data,
                    "diagnosis": diagnosis,
                    "treatment": treatment,
                    "actions_taken": actions_taken,
                    "escalated": escalated
                }

            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": "Continue to the next step."})

        return {
            "status": "complete" if not escalated else "escalated",
            "reasoning_trace": reasoning_trace,
            "diagnosis": diagnosis,
            "treatment": treatment,
            "actions_taken": actions_taken,
            "hallucination_flags": [],
            "escalated": escalated
        }

    def resume_after_tool(self, messages, tool_name, tool_result,
                          reasoning_trace, process_state,
                          diagnosis, treatment, actions_taken, escalated) -> dict:
        # inject the tool result and keep going
        messages.append({"role": "user", "content": f"'{tool_name}' returned: {tool_result}\nContinue."})
        if reasoning_trace:
            last = reasoning_trace[-1]
            if isinstance(last, dict):
                last['observation'] = tool_result
            else:
                last.observation = tool_result

        _extra = _llamacpp_extra(self.provider, self.model)
        for step_num in range(len(reasoning_trace) + 1, 20):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                **_extra
            )
            self.total_tokens += resp.usage.total_tokens
            raw = resp.choices[0].message.content

            try:
                step_data = _parse_json_response(raw, self.provider)
            except json.JSONDecodeError:
                break

            step = ReasoningStep(
                step=step_num,
                thought=step_data.get("thought", ""),
                action=step_data.get("action"),
                observation=None,
                confidence=step_data.get("confidence", 1.0),
                gateway_decision=step_data.get("gateway_decision")
            )

            if step_data.get("confidence", 1.0) < self.confidence_threshold:
                escalated = True
                reasoning_trace.append(step)
                break

            if step_data.get("diagnosis"):
                diagnosis = step_data["diagnosis"]
            if step_data.get("treatment"):
                treatment = step_data["treatment"]
            if step_data.get("action"):
                actions_taken.append(step_data["action"])

            reasoning_trace.append(step)

            if step_data.get("process_complete"):
                break

            if step_data.get("requires_tool"):
                return {
                    "status": "tool_required",
                    "tool_name": step_data.get("tool_name"),
                    "tool_params": step_data.get("tool_params", {}),
                    "reasoning_trace": reasoning_trace,
                    "process_state": process_state,
                    "messages": messages,
                    "step_data": step_data,
                    "diagnosis": diagnosis,
                    "treatment": treatment,
                    "actions_taken": actions_taken,
                    "escalated": escalated
                }

            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": "Next step."})

        return {
            "status": "complete" if not escalated else "escalated",
            "reasoning_trace": reasoning_trace,
            "diagnosis": diagnosis,
            "treatment": treatment,
            "actions_taken": actions_taken,
            "hallucination_flags": [],
            "escalated": escalated
        }
