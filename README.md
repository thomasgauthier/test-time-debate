# Swarm Intelligence Congress

LLM implementation of Lewis Rosenberg's Conversational Swarm Intelligence (CSI). Replaces human participants with LLM agents representing source documents to reach collective superintelligence.

## Architecture
1. **Parallel Congresses**: Documents are partitioned into small groups (default: 5) for local deliberation.
2. **Global Reduction**: `SwarmIntelligenceReducer` extracts cross-congress insights from initial transcripts.
3. **Intelligence Briefing**: `InsightReporter` feeds global context back to local groups as "Briefings."
4. **Recursive Deliberation**: Groups run a second round of debate informed by the global state.
5. **Synthesis**: `CollectiveIntelligenceSynthetizer` generates a final answer from all enhanced transcripts.

## Core Components
- `conversation_orchestrator.py`: Async graph execution and process management.
- `conversational_agent.py`: DSPy-based agent logic for document representatives.
- `swarm_intelligence_reducer.py`: Global insight extractor (Map phase).
- `insight_reporter.py`: Local context injector (Shuffle/Inject phase).
- `collective_intelligence_synthetizer.py`: Final result generator (Reduce phase).

## Setup
1. **Data**: Place source documents in `all_documents_chris_lee.json`.
2. **Env**: Configure `.env` with API keys and Langfuse credentials.
3. **LM**: Update `lm_kwargs` in `conversation_orchestrator.py` for your endpoint.
4. **Execution**:
   ```bash
   python conversation_orchestrator.py
   ```

## Dependencies
`dspy`, `langfuse`, `openinference-instrumentation-dspy`, `tqdm`, `rich`.
