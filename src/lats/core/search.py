"""Language Agent Tree Search implementation.

This module contains the main LanguageAgentTreeSearch class that orchestrates
the MCTS-based search with LLM reflection.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from lats.core.policies import Decision, select_leaf, should_continue
from lats.core.scoring import compute_combined_reward, compute_self_consistency
from lats.models.config import LATSConfig
from lats.models.node import SearchNode
from lats.models.reflection import Reflection
from lats.models.state import TreeState


class LanguageAgentTreeSearch:
    """Language Agent Tree Search using Monte Carlo Tree Search with LLM reflection.

    LATS combines MCTS with language model self-reflection to navigate a search
    space of candidate responses. At each step:
    1. Select a leaf node using UCT (Upper Confidence Bound for Trees)
    2. Generate multiple candidate responses
    3. Evaluate each candidate with self-reflection
    4. Backpropagate scores to update node values
    5. Repeat until solution found or max depth reached

    Attributes:
        config: Algorithm configuration parameters
        llm: Language model for generation and reflection
        tools: List of tools available to the agent
        tool_node: LangGraph tool execution node
        prompt_template: Template for candidate generation
        reflection_chain: Chain for reflection/evaluation
        initial_answer_chain: Chain for initial candidate
        graph: Compiled LangGraph execution graph

    Example:
        >>> config = LATSConfig(model="gpt-4o", n_candidates=5, max_depth=5)
        >>> lats = LanguageAgentTreeSearch(config=config)
        >>> solution, trajectory = lats.run("What is Python?")
        >>> print(solution.reflection.score)
        9
    """

    def __init__(
        self,
        config: LATSConfig | None = None,
        llm_factory: Callable[[str], ChatOpenAI] | None = None,
        tool_node: ToolNode | None = None,
    ) -> None:
        """Initialize the LATS instance.

        Args:
            config: Algorithm configuration (uses defaults if None)
            llm_factory: Optional factory for creating LLM instances.
                        Useful for testing with mock LLMs.
            tool_node: Optional pre-configured tool node.
                      If None, creates default Tavily search tool.

        Raises:
            LATSConfigError: If configuration is invalid
        """
        self.config = config or LATSConfig()
        self.config.validate()

        llm_builder = llm_factory or (lambda model: ChatOpenAI(model=model))
        self.llm = llm_builder(self.config.model)

        if tool_node is None:
            search = TavilySearchAPIWrapper()
            tavily_tool = TavilySearchResults(
                api_wrapper=search,
                max_results=self.config.max_search_results,
            )
            self.tools = [tavily_tool]
            self.tool_node = ToolNode(tools=self.tools)
        else:
            self.tool_node = tool_node
            self.tools: list[Any] = []

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an AI assistant."),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="messages", optional=True),
            ]
        )

        reflection_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Reflect on and grade the assistant response to the user question below.\n\n"
                    "Score the response on three dimensions (each 0-10):\n\n"
                    "1. **Evidence quality** — Are the retrieved artefacts (logs, metrics, "
                    "traces) relevant to the failure and sufficient to support the "
                    "diagnosis? Award high scores when evidence directly addresses the "
                    "failure mode; penalise when critical data sources are ignored or "
                    "irrelevant artefacts are cited.\n\n"
                    "2. **Diagnostic completeness** — Does the response enumerate "
                    "plausible root-cause hypotheses and systematically confirm or rule "
                    "out each one? Award high scores for methodical elimination; penalise "
                    "when obvious hypotheses are omitted or only a single cause is "
                    "considered.\n\n"
                    "3. **Internal consistency** — Are the claims logically coherent, "
                    "free of contradictions, and properly supported by the cited evidence? "
                    "Penalise when conclusions contradict the presented data or when "
                    "reasoning steps are missing.",
                ),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="candidate"),
            ]
        )

        self.reflection_chain = (
            reflection_prompt
            | self.llm.bind_tools(tools=[Reflection], tool_choice="Reflection").with_config(
                run_name="Reflection"
            )
            | PydanticToolsParser(tools=[Reflection])
        )

        self.initial_answer_chain = self.prompt_template | self.llm.bind_tools(
            tools=self.tools
        ).with_config(run_name="GenerateInitialCandidate")

        self.graph = self._build_graph()

    def _reflect(self, question: str, candidate: list[BaseMessage]) -> Reflection:
        """Generate reflection for a single candidate response.

        Args:
            question: Original user question
            candidate: Candidate response messages

        Returns:
            Reflection containing score and critique

        Example:
            >>> reflection = lats._reflect("What is Python?", [AIMessage(...)])
            >>> reflection.score
            8
        """
        parsed = self.reflection_chain.invoke({"input": question, "candidate": candidate})
        reflection = parsed[0]

        if not isinstance(candidate[-1], AIMessage):
            reflection.found_solution = False

        return reflection

    def _reflect_batch(
        self, question: str, candidates: list[list[BaseMessage]]
    ) -> list[Reflection]:
        """Generate reflections for multiple candidates in batch.

        Batch processing is more efficient than individual reflection calls.

        Args:
            question: Original user question
            candidates: List of candidate response message lists

        Returns:
            List of reflections, one per candidate

        Example:
            >>> candidates = [[AIMessage(...)], [AIMessage(...)]]
            >>> reflections = lats._reflect_batch("What is Python?", candidates)
            >>> len(reflections)
            2
        """
        parsed = self.reflection_chain.batch(
            [{"input": question, "candidate": candidate} for candidate in candidates]
        )

        reflections: list[Reflection] = []
        for candidate, reflection_list in zip(candidates, parsed, strict=True):
            reflection = reflection_list[0]

            if not isinstance(candidate[-1], AIMessage):
                reflection.found_solution = False

            reflections.append(reflection)

        return reflections

    @staticmethod
    def _to_tool_message_input(tool_call: Any) -> dict[str, list[AIMessage]]:
        """Convert a tool call to LangGraph tool node input format.

        Args:
            tool_call: Tool call dictionary or object from AIMessage

        Returns:
            Formatted input for ToolNode

        Example:
            >>> tool_call = {"name": "search", "args": {"query": "Python"}, "id": "1"}
            >>> input_dict = LanguageAgentTreeSearch._to_tool_message_input(tool_call)
            >>> input_dict.keys()
            dict_keys(['messages'])
        """
        if isinstance(tool_call, dict):
            name = tool_call["name"]
            args = tool_call["args"]
            tool_id = tool_call["id"]
        else:
            name = tool_call.get("name") or getattr(tool_call, "name", "")
            args = tool_call.get("args") or getattr(tool_call, "args", {})
            tool_id = tool_call.get("id") or getattr(tool_call, "id", "")

        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": name,
                            "args": args,
                            "id": tool_id,
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        }

    def _run_tools_for_candidate(self, candidate: AIMessage) -> list[BaseMessage]:
        """Execute tool calls from a candidate response.

        Args:
            candidate: Candidate AIMessage potentially containing tool calls

        Returns:
            List of tool response messages

        Example:
            >>> candidate = AIMessage(content="", tool_calls=[...])
            >>> tool_responses = lats._run_tools_for_candidate(candidate)
            >>> # Returns ToolMessage instances
        """
        if not candidate.tool_calls:
            return []

        tool_inputs = [self._to_tool_message_input(tc) for tc in candidate.tool_calls]
        tool_outputs = self.tool_node.batch(tool_inputs)

        return [output["messages"][-1] for output in tool_outputs]

    def _generate_candidates(
        self, question: str, trajectory: list[BaseMessage]
    ) -> list[AIMessage]:
        """Generate multiple candidate responses for a given trajectory.

        Uses the LLM's n parameter to generate multiple candidates in one call.

        Args:
            question: Original user question
            trajectory: Message history up to this point

        Returns:
            List of candidate AIMessages

        Example:
            >>> candidates = lats._generate_candidates("What is Python?", trajectory)
            >>> len(candidates)  # Equals config.n_candidates
            5
        """
        prompt = self.prompt_template.invoke({"input": question, "messages": trajectory})
        bound_llm = self.llm.bind_tools(tools=self.tools)

        result = self.llm.generate(
            [prompt.to_messages()],
            n=self.config.n_candidates,
            run_name="GenerateCandidates",
            **bound_llm.kwargs,  # type: ignore[attr-defined]
        )

        return [generation.message for generation in result.generations[0]]  # type: ignore[attr-defined,misc]

    def _start(self, state: TreeState) -> TreeState:
        """Initialize the search tree with the first candidate.

        This is the START node in the LangGraph execution graph.

        Args:
            state: Initial tree state with user input

        Returns:
            Updated state with root node containing first candidate

        Example:
            >>> state = {"input": "What is Python?", "root": None}
            >>> new_state = lats._start(state)
            >>> new_state["root"].depth
            1
        """
        candidate = self.initial_answer_chain.invoke({"input": state["input"]})

        messages: list[BaseMessage] = [candidate]
        if isinstance(candidate, AIMessage):
            tool_messages = self._run_tools_for_candidate(candidate)
            messages.extend(tool_messages)

        reflection = self._reflect(state["input"], messages)

        return {**state, "root": SearchNode(messages=messages, reflection=reflection)}

    def _expand(self, state: TreeState) -> TreeState:
        """Expand the search tree by selecting and expanding a leaf node.

        This is the EXPAND node in the LangGraph execution graph.

        The expansion process:
        1. Select best leaf using UCT
        2. Generate n candidates from that leaf
        3. Execute tool calls for each candidate
        4. Reflect on all candidates
        5. Add candidates as children of the selected leaf

        Args:
            state: Current tree state

        Returns:
            Updated state with expanded tree

        Example:
            >>> state = lats._expand(state)
            >>> # Tree has been expanded with new nodes
        """
        best_leaf = select_leaf(state["root"], exploration_weight=self.config.exploration_weight)
        trajectory = best_leaf.get_trajectory()

        candidates = self._generate_candidates(state["input"], trajectory)

        batched_messages: list[list[BaseMessage]] = [[candidate] for candidate in candidates]
        flat_tool_inputs: list[dict[str, list[AIMessage]]] = []
        flat_candidate_index: list[int] = []

        for index, candidate in enumerate(candidates):
            for tool_call in candidate.tool_calls:
                flat_tool_inputs.append(self._to_tool_message_input(tool_call))
                flat_candidate_index.append(index)

        if flat_tool_inputs:
            tool_outputs = self.tool_node.batch(flat_tool_inputs)
            for candidate_index, output in zip(flat_candidate_index, tool_outputs, strict=True):
                batched_messages[candidate_index].append(output["messages"][-1])

        reflections = self._reflect_batch(state["input"], batched_messages)
        consistency_scores = compute_self_consistency(candidates)

        children = [
            SearchNode(
                messages=messages,
                reflection=reflection,
                parent=best_leaf,
                reward_override=compute_combined_reward(
                    reflection_score=reflection.normalized_score,
                    self_consistency=consistency,
                    alpha=self.config.consistency_weight,
                ),
            )
            for messages, reflection, consistency in zip(
                batched_messages, reflections, consistency_scores, strict=True
            )
        ]

        best_leaf.children.extend(children)

        return state

    def _route(self, state: TreeState) -> Decision:
        """Determine whether to continue expanding or terminate.

        This implements the conditional edge logic in the LangGraph.

        Args:
            state: Current tree state

        Returns:
            "expand" to continue, or END to terminate

        Example:
            >>> decision = lats._route(state)
            >>> if decision == "expand":
            ...     # Continue search
        """
        return should_continue(state["root"], max_depth=self.config.max_depth)

    def _build_graph(self) -> Any:
        """Build the LangGraph execution graph.

        The graph structure:
        START -> start -> [expand | END]
                          expand -> [expand | END]

        Returns:
            Compiled LangGraph

        Example:
            >>> graph = lats._build_graph()
            >>> # Can now execute with graph.stream()
        """
        graph = StateGraph(TreeState)

        graph.add_node("start", self._start)
        graph.add_node("expand", self._expand)

        graph.add_edge(START, "start")
        graph.add_conditional_edges("start", self._route, ["expand", END])
        graph.add_conditional_edges("expand", self._route, ["expand", END])

        return graph.compile()

    def run(
        self, question: str, print_rollouts: bool = False
    ) -> tuple[SearchNode, list[BaseMessage]]:
        """Execute LATS search for a given question.

        Args:
            question: User's input question or task
            print_rollouts: Whether to print tree depth after each expansion

        Returns:
            Tuple of (best_solution_node, solution_trajectory)

        Raises:
            RuntimeError: If graph execution fails or returns no steps

        Example:
            >>> lats = LanguageAgentTreeSearch()
            >>> solution, trajectory = lats.run("What is Python?", print_rollouts=True)
            start: rolled out depth=1
            expand: rolled out depth=2
            expand: rolled out depth=3
            >>> solution.reflection.score
            9
            >>> len(trajectory)
            # Messages from root to solution
        """
        last_step: dict[str, TreeState] | None = None

        for step in self.graph.stream({"input": question}):
            last_step = step

            if print_rollouts:
                name, step_state = next(iter(step.items()))
                print(f"{name}: rolled out depth={step_state['root'].height}")

        if last_step is None:
            raise RuntimeError("Graph execution returned no steps")

        final_state = next(iter(last_step.values()))

        solution = final_state["root"].get_best_solution()
        trajectory = solution.get_trajectory(include_reflections=False)

        return solution, trajectory
