from __future__ import annotations

try:
    from langgraph.graph import END, StateGraph
except ImportError:  # pragma: no cover - bootstrap fallback
    END = "__end__"
    StateGraph = None

from advisor.config import Settings
from advisor.knowledge_base import format_context_snippets, load_market_documents
from advisor.model_service import ModelRegistry
from advisor.reasoning import InvestmentReasoner
from advisor.reporting import ReportBuilder
from advisor.retriever import MarketRetriever
from advisor.schemas import AgentState, PropertyInput, RetrievedContext


class RealEstateAdvisorAgent:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.load()
        documents = load_market_documents(self.settings.knowledge_base_path)
        self.model_registry = ModelRegistry(self.settings.artifacts_dir)
        self.retriever = MarketRetriever(documents)
        self.reasoner = InvestmentReasoner(
            undervalued_threshold=self.settings.undervalued_threshold,
            overvalued_threshold=self.settings.overvalued_threshold,
        )
        self.report_builder = ReportBuilder()
        self.graph = self._build_graph().compile() if StateGraph is not None else None

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        workflow.add_node("input_processing", self._input_processing_node)
        workflow.add_node("price_prediction", self._price_prediction_node)
        workflow.add_node("market_retrieval", self._market_retrieval_node)
        workflow.add_node("reasoning", self._reasoning_node)
        workflow.add_node("report_generation", self._report_generation_node)

        workflow.set_entry_point("input_processing")
        workflow.add_edge("input_processing", "price_prediction")
        workflow.add_edge("price_prediction", "market_retrieval")
        workflow.add_edge("market_retrieval", "reasoning")
        workflow.add_edge("reasoning", "report_generation")
        workflow.add_edge("report_generation", END)
        return workflow

    def run(self, raw_input: dict) -> dict:
        if self.graph is None:
            result = self._run_without_langgraph(raw_input)
        else:
            result = self.graph.invoke({"user_input": raw_input})
        return result["final_report"]

    def _run_without_langgraph(self, raw_input: dict) -> AgentState:
        state: AgentState = {"user_input": raw_input}
        for node in (
            self._input_processing_node,
            self._price_prediction_node,
            self._market_retrieval_node,
            self._reasoning_node,
            self._report_generation_node,
        ):
            state.update(node(state))
        return state

    def _input_processing_node(self, state: AgentState) -> AgentState:
        user_input = PropertyInput.model_validate(state["user_input"])
        return {
            "normalized_input": user_input.model_dump(),
        }

    def _price_prediction_node(self, state: AgentState) -> AgentState:
        user_input = PropertyInput.model_validate(state["normalized_input"])
        predicted_price, model_predictions = self.model_registry.predict(
            user_input,
            self.settings.default_model_name,
        )
        return {
            "predicted_price": predicted_price,
            "selected_model": self.settings.default_model_name,
            "model_predictions": model_predictions,
        }

    def _market_retrieval_node(self, state: AgentState) -> AgentState:
        user_input = PropertyInput.model_validate(state["normalized_input"])
        items = self.retriever.search(user_input, self.settings.retrieval_top_k)
        return {
            "retrieved_context": [item.model_dump() for item in items],
            "market_summary": format_context_snippets(items),
        }

    def _reasoning_node(self, state: AgentState) -> AgentState:
        user_input = PropertyInput.model_validate(state["normalized_input"])
        contexts = [
            RetrievedContext.model_validate(item) for item in state.get("retrieved_context", [])
        ]
        analysis = self.reasoner.analyze(
            user_input=user_input,
            predicted_price=state["predicted_price"],
            contexts=contexts,
        )
        return analysis

    def _report_generation_node(self, state: AgentState) -> AgentState:
        user_input = PropertyInput.model_validate(state["normalized_input"])
        payload = {
            "predicted_price": state["predicted_price"],
            "selected_model": state["selected_model"],
            "market_summary": state["market_summary"],
            "investment_score": state["investment_score"],
            "risk_level": state["risk_level"],
            "recommendation": state["recommendation"],
            "justification": state["justification"],
            "valuation_gap_pct": state["valuation_gap_pct"],
            "budget_fit": state["budget_fit"],
            "comparable_market_price": state["comparable_market_price"],
            "confidence_note": state["confidence_note"],
        }
        report = self.report_builder.build(user_input, payload)
        return {
            "final_report": report.model_dump(),
        }
