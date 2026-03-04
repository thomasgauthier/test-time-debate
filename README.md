# Document Swarm Intelligence Congress

A document-deliberation system inspired by [Conversational Swarm Intelligence](https://arxiv.org/pdf/2309.12366) (CSI). Documents are represented by AI agents that debate and synthesize collective knowledge.

## Overview

This system simulates a congress where each document has an AI representative. Representatives debate questions in small groups from their document's perspective. Transcripts from all debates are centrally processed to extract insights, which are then redistributed to each group to inform a second round of debate. This pipeline produces collective intelligence output.

## Architecture

The system follows a 5-stage pipeline:

1. **Parallel Congresses** - Documents are partitioned into groups of ~5. Each group runs a debate where document representatives argue what their source would say about the query.

2. **Global Reduction** - All transcripts from all congresses are processed together to extract cross-congress insights using a Chain-Of-Agent or MapReduce pattern.

3. **Intelligence Reporting** - Group-specific briefings are generated from global insights and injected back into congress sessions.

4. **Recursive Deliberation** - Congresses run a second debate round, now informed by intelligence from other sessions.

5. **Synthesis** - All enhanced transcripts are synthesized into a final collective answer using Chain-Of-Agent.

## Requirements

- Python 3.12+
- A local LLM endpoint (defaults to `http://localhost:9292/v1`)
- Langfuse instance (optional, for observability)

## Installation

```bash
uv sync
```

## Configuration

Set environment variables:

```
LANGFUSE_PUBLIC_KEY=your_key
LANGFUSE_SECRET_KEY=your_key
LANGFUSE_HOST=your_host
```

Configure the LLM endpoint in `conversation_orchestrator.py`:

```python
lm_kwargs = {
    "api_key": "hello",
    "api_base": "http://localhost:9292/v1",
    "model": "openai/qwen3.5-35b-a3b",
}
```

## Data

Place documents in `all_documents.json`:

```json
[
  {"title": "Document 1", "content": "..."},
  {"title": "Document 2", "content": "..."}
]
```

## Usage

```bash
python main.py
```

Enter questions interactively. Results are saved to directories named by query hash.

## Output Structure

Each run creates a directory containing:

- `question.txt` - Original query
- `congress_results.txt` - Initial debate transcripts
- `reports.txt` - Intelligence briefings
- `conversations_with_briefings.txt` - Second round transcripts
- `final_collective_answer.txt` - Synthesized answer

## Key Components

| File | Purpose |
|------|---------|
| `main.py` | Interactive REPL entry point |
| `conversation_orchestrator.py` | Core orchestration and transcript management |
| `conversational_agent.py` | Document representative implementation |
| `swarm_intelligence_reducer.py` | Cross-congress insight extraction |
| `insight_reporter.py` | Intelligence briefing generation |
| `collective_intelligence_synthetizer.py` | Final answer synthesis |

## Background

This draws inspiration from Conversational Swarm Intelligence (CSI) research by Louis Rosenberg, applying similar principles to document collections rather than human groups.

This system loosely applies the CSI concepts. Unlike retrieval augmented Q&A systems that use search, here each data point is seen during generation. Documents are represented by AI agents that deliberate in small groups. However, unlike biological swarms where information flows through local interactions, this system uses centralized processing:

- Local debates happen in parallel groups
- Transcripts are processed in chunks (~10 at a time) using Chain-of-Agents: insights accumulate sequentially across chunks
- Group-specific briefings are generated from those insights and distributed to each congress

This MapReduce-style architecture processes information differently than distributed biological systems, but still enables collective intelligence through multi-round deliberation.

## License

GPL-3.0
