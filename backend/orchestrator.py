import uuid
from typing import TypedDict, Optional, List, Literal
from langgraph.graph import StateGraph, END
from agents.knowledge_agent import KnowledgeAgent
from agents.reasoning_agent import ReasoningAgent
from agents.execution_agent import ExecutionAgent
from database.faiss_store import FAISSRuleStore
from models.schemas import ProcessRequest, ProcessResult, RuleMatch, ReasoningStep
from config.settings import settings, AblationMode, Provider


# LangGraph state definition
# everything the graph needs to pass between nodes lives here

class TriAgentState(TypedDict):
    process_id: str
    input_description: str
    context: dict
    rule_data: Optional[dict]
    reasoning_result: Optional[dict]
    tool_name: Optional[str]
    tool_params: Optional[dict]
    tool_result: Optional[str]
    messages: List[dict]
    actions_taken: List[str]
    reasoning_trace: List[dict]
    diagnosis: Optional[str]
    treatment: Optional[str]
    escalated: bool
    complete: bool
    tool_iterations: int
    error: Optional[str]


class TriAgentOrchestrator:

    def __init__(self, model: str = "gpt-4o", rules_path: str = "data/rules",
                 provider: Provider = None, ablation_mode: AblationMode = "full",
                 skip_prefetch: bool = False):
        self.model = model
        self.provider = provider or settings.backbone_provider
        self.ablation_mode = ablation_mode
        self.rules_path = rules_path
        self.skip_prefetch = skip_prefetch

        if ablation_mode != "no_knowledge":
            self.rule_store = FAISSRuleStore()
            self.rule_store.load(rules_path)
        else:
            self.rule_store = None
        self.knowledge_agent = KnowledgeAgent(self.rule_store, model=model, provider=self.provider)
        self.reasoning_agent = ReasoningAgent(model=model, provider=self.provider)
        self.execution_agent = ExecutionAgent()

        # build the LangGraph workflow
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(TriAgentState)

        # register nodes
        workflow.add_node("knowledge", self._knowledge_node)
        workflow.add_node("reasoning", self._reasoning_node)
        workflow.add_node("execution", self._execution_node)

        # entry point
        workflow.set_entry_point("knowledge")

        # knowledge always goes to reasoning
        workflow.add_edge("knowledge", "reasoning")

        # reasoning decides what comes next
        workflow.add_conditional_edges(
            "reasoning",
            self._routing,
            {
                "execute": "execution",
                "complete": END,
                "escalate": END
            }
        )

        # after execution, go back to reasoning
        workflow.add_edge("execution", "reasoning")

        return workflow.compile()

    # nodes

    def _knowledge_node(self, state: TriAgentState) -> TriAgentState:
        if self.ablation_mode == "no_knowledge":
            # bypass: pass raw description as the "rule" so reasoning has no structured knowledge
            rule_data = {
                "rule_id": "ablation_no_knowledge",
                "trigger": state["input_description"][:200],
                "content": state["input_description"],
                "similarity_score": 0.0,
                "source": "ablation_bypass",
                "metadata": {},
            }
            return {**state, "rule_data": rule_data}
        rule_data = self.knowledge_agent.handle_knowledge_request(
            state["input_description"]
        )
        return {**state, "rule_data": rule_data}

    def _reasoning_node(self, state: TriAgentState) -> TriAgentState:
        # first pass: no tool result yet
        if state.get("tool_result") is None and not state.get("messages"):
            result = self.reasoning_agent.run(
                process_rule=state["rule_data"]["content"],
                input_description=state["input_description"],
                context=state["context"]
            )
        else:
            # resume after a tool call
            result = self.reasoning_agent.resume_after_tool(
                messages=state["messages"],
                tool_name=state["tool_name"],
                tool_result=state["tool_result"],
                reasoning_trace=state["reasoning_trace"],
                process_state={},
                diagnosis=state.get("diagnosis"),
                treatment=state.get("treatment"),
                actions_taken=state.get("actions_taken", []),
                escalated=state.get("escalated", False)
            )

        return {
            **state,
            "reasoning_result": result,
            "messages": result.get("messages", state.get("messages", [])),
            "reasoning_trace": [s.__dict__ if hasattr(s, '__dict__') else s
                                 for s in result.get("reasoning_trace", [])],
            "diagnosis": result.get("diagnosis") or state.get("diagnosis"),
            "treatment": result.get("treatment") or state.get("treatment"),
            "actions_taken": result.get("actions_taken", state.get("actions_taken", [])),
            "escalated": result.get("escalated", False),
            "complete": result.get("status") == "complete",
            "tool_name": result.get("tool_name"),
            "tool_params": result.get("tool_params"),
            # reset tool result so next reasoning pass starts fresh
            "tool_result": None
        }

    def _execution_node(self, state: TriAgentState) -> TriAgentState:
        if self.ablation_mode == "no_execution":
            # bypass: return a fixed stub so reasoning must conclude on parametric knowledge
            return {
                **state,
                "tool_result": "TOOL_DISABLED_FOR_ABLATION",
                "tool_iterations": state.get("tool_iterations", 0) + 1,
            }

        tool_result = self.execution_agent.execute(
            state["tool_name"],
            state["tool_params"] or {}
        )

        if tool_result.get("escalate"):
            return {**state, "escalated": True, "complete": True}

        return {
            **state,
            "tool_result": str(tool_result.get("result", tool_result.get("error", ""))),
            "tool_iterations": state.get("tool_iterations", 0) + 1
        }

    # routing logic

    def _routing(self, state: TriAgentState) -> str:
        if state.get("escalated"):
            return "escalate"
        if state.get("complete"):
            return "complete"
        if state.get("tool_iterations", 0) >= 10:
            return "escalate"
        result = state.get("reasoning_result", {})
        if result.get("status") == "tool_required":
            return "execute"
        return "complete"

    # public API

    def process(self, request: ProcessRequest) -> ProcessResult:
        process_id = str(uuid.uuid4())[:8]

        # build context
        context = {k: v for k, v in {
            "crop": request.crop,
            "location": request.location,
            "growth_stage": request.growth_stage,
            "severity": request.severity
        }.items() if v is not None}

        # pull live weather if location given (skip in no_execution ablation and tool-forcing mode)
        if request.location and self.ablation_mode != "no_execution" and not self.skip_prefetch:
            weather = self.execution_agent.execute("get_weather", {"location": request.location})
            if weather["success"]:
                context["weather"] = weather["result"]

        # initial state
        initial_state: TriAgentState = {
            "process_id": process_id,
            "input_description": request.description,
            "context": context,
            "rule_data": None,
            "reasoning_result": None,
            "tool_name": None,
            "tool_params": None,
            "tool_result": None,
            "messages": [],
            "actions_taken": [],
            "reasoning_trace": [],
            "diagnosis": None,
            "treatment": None,
            "escalated": False,
            "complete": False,
            "tool_iterations": 0,
            "error": None
        }

        # run the graph
        final_state = self.graph.invoke(initial_state)

        # build rule match
        rule_data = final_state.get("rule_data", {})
        rule_match = RuleMatch(
            rule_id=rule_data.get("rule_id", ""),
            trigger=rule_data.get("trigger", ""),
            content=rule_data.get("content", ""),
            similarity_score=rule_data.get("similarity_score", 0.0)
        )

        # compute cost using per-model rates; local models return (0,0)
        input_rate, _ = settings.cost_table.get(self.model, (2.50, 10.00))
        total_tokens = (
            self.knowledge_agent.total_tokens +
            self.reasoning_agent.total_tokens
        )
        cost = round((total_tokens / 1_000_000) * input_rate, 4)

        # rebuild reasoning trace as ReasoningStep objects
        trace = []
        for s in final_state.get("reasoning_trace", []):
            if isinstance(s, dict):
                trace.append(ReasoningStep(**s))
            else:
                trace.append(s)

        return ProcessResult(
            success=final_state.get("complete", False) and not final_state.get("escalated", False),
            process_id=process_id,
            diagnosis=final_state.get("diagnosis"),
            treatment=final_state.get("treatment"),
            actions_taken=final_state.get("actions_taken", []),
            reasoning_trace=trace,
            rule_used=rule_match,
            hallucination_flags=[],
            escalated=final_state.get("escalated", False),
            cost_usd=cost,
            llm_model=self.model
        )

    def load_documents(self, documents_path: str):
        self.knowledge_agent.load_fao_documents(documents_path)
        self.rule_store.save(self.rules_path)

    def get_rules_count(self) -> int:
        return len(self.rule_store)
