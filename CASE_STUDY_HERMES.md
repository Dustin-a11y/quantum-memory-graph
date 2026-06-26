# Case Study: How Hermes Agent Uses QMG

## Overview

[Hermes Agent](https://hermes-agent.nousresearch.com) is an open-source AI agent framework by Nous Research. It powers a fleet of 12+ specialized agents — DK, Kamek, Daisy, Wario, Bowser, Birdo, Rosalina, Koopa, Mario, Luigi, Toadstool, and Yoshi — each designed for distinct workflows and personality profiles. Across this fleet, agents handle thousands of conversations and need to remember user preferences, environment facts, and project decisions across sessions.

## The Challenge

Hermes agents need memory that goes beyond simple keyword or vector search. When a user mentions they're building a React app with FastAPI, the agent needs to recall not just those two facts, but *all related context together* — database choices, deployment preferences, team decisions — as a connected whole, not as isolated search results. Flat vector databases return relevant documents but lose the relationships between them.

## Architecture

Each agent runs its own QMG instance with an isolated knowledge graph. Memories are stored locally and never shared between agents, ensuring clean separation of context:

```
┌─────────────────────────────────────────────────────────────┐
│                      Hermes Agent Fleet                      │
│                                                              │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐      ┌──────┐     │
│  │  DK  │  │Kamek │  │Daisy │  │Wario │ ...  │Yoshi │     │
│  │ :8400│  │ :8401│  │ :8402│  │ :8403│      │ :8411│     │
│  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘      └──┬───┘     │
│     │         │         │         │               │         │
│     ▼         ▼         ▼         ▼               ▼         │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐      ┌──────┐     │
│  │ QMG  │  │ QMG  │  │ QMG  │  │ QMG  │      │ QMG  │     │
│  │Graph │  │Graph │  │Graph │  │Graph │ ...  │Graph │     │
│  └──────┘  └──────┘  └──────┘  └──────┘      └──────┘     │
│                                                              │
│   Each agent runs a local QMG server on its own port.        │
│   Memories auto-store during conversation.                   │
│   Recall queries the graph for connected, relevant memories. │
└─────────────────────────────────────────────────────────────┘
```

## How It Works

### 1. Auto-Store via SOUL.md

Each Hermes agent is governed by a `SOUL.md` configuration file that mandates automatic memory storage after high-signal responses. When an agent produces a response containing important information — user preferences, technical decisions, environment facts — the memory is automatically stored in QMG without any explicit user action:

```
store("User prefers Poetry over pip for Python package management")
store("Project Phoenix uses Next.js 14 with App Router")
store("Deployment target is AWS ECS with Fargate")
```

### 2. Relationship-Aware Storage

QMG doesn't just store memories — it builds a knowledge graph. Each memory becomes a node, and QMG automatically creates weighted edges based on:

- **Semantic similarity** between memory contents
- **Entity co-occurrence** (projects, tools, people mentioned together)
- **Temporal proximity** (memories stored close together in time)
- **Source tracking** (memories from the same conversation or document)

### 3. QAOA-Optimized Retrieval

When an agent needs to recall context, QMG uses graph traversal combined with QAOA (Quantum Approximate Optimization Algorithm) subgraph optimization. This finds the optimal *set* of memories — not just the top-K by similarity, but the set that maximizes both relevance and internal connectivity:

```python
results = recall("What's Project Phoenix's deployment setup?", K=5)
# Returns: Next.js, ECS Fargate, AWS config, CI/CD pipeline
# — all connected, all relevant, no noise
```

Noise memories (e.g., "team had pizza for lunch") are automatically excluded because they lack graph connections to the query's semantic cluster, even if they share keywords.

## Results

Across the Hermes agent fleet, QMG powers persistent, relationship-aware memory:

| Metric | Value |
|--------|-------|
| Total nodes across fleet | 373 |
| Total edges across fleet | 11,523 |
| Average degree | 61.8 |
| LongMemEval NDCG@10 | 94.26% (#1) |
| R@1 accuracy | 90.6% |

Agents remember **user preferences** (preferred tools, workflows, naming conventions), **environment facts** (installed packages, system configurations, API keys), and **project decisions** (architecture choices, tech stack, deployment targets) across sessions. QAOA subgraph selection ensures that when an agent recalls memories about a specific topic, it gets the complete connected picture — not just a list of similar-sounding facts.

## Key Takeaways

1. **Isolated graphs per agent** prevent context leakage and keep each agent's memory clean
2. **Auto-store via SOUL.md** makes memory capture zero-friction — no manual `store()` calls needed
3. **Relationship-aware retrieval** means agents understand *connections*, not just similarity
4. **QAOA optimization** (optional, works classically too) finds the optimal subgraph for complex recall
5. **MIT license** fits the open-source Hermes ecosystem
