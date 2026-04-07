"""
MemCombine Benchmark — Tests memory COMBINATION quality.

Unlike LongMemEval (needle-in-haystack retrieval), MemCombine tests whether
selected memories work TOGETHER to answer complex questions.

Questions require synthesizing information from multiple memories:
  - "What was the decision AND its reasoning AND its outcome?"
  - "How do project X and project Y relate?"
  - "What changed between meeting A and meeting B?"

Metrics:
  - Combination Score: Do selected memories cover all required facets?
  - Synergy Score: Do memories reference/build on each other?
  - Completeness: Can the question be fully answered from selected memories?

Copyright 2026 Coinkong (Chef's Attraction). MIT License.
"""

import json
import numpy as np
from typing import List, Dict
from dataclasses import dataclass, field


@dataclass
class MemCombineQuestion:
    """A question requiring multiple related memories."""
    id: str
    question: str
    category: str  # synthesis, temporal, causal, multi-entity
    memories: List[Dict]  # All available memories
    evidence_ids: List[int]  # Which memories contain evidence
    facets: List[str]  # Required information facets
    facet_memory_map: Dict  # Which facet comes from which memory


# Built-in benchmark scenarios
SCENARIOS = [
    {
        "id": "synthesis_1",
        "question": "What technology stack was chosen for the project and why was each component selected?",
        "category": "synthesis",
        "memories": [
            {"id": 0, "text": "Team meeting: Decided to use React for the frontend. Sarah argued it has the best ecosystem for our use case."},
            {"id": 1, "text": "Architecture review: PostgreSQL chosen for the database. Need JSONB support for flexible schemas."},
            {"id": 2, "text": "Sprint planning: Set up CI/CD pipeline using GitHub Actions. Two-week sprint cycles."},
            {"id": 3, "text": "Team lunch at the Italian place. Good pasta. Bob told a funny joke about recursion."},
            {"id": 4, "text": "Backend discussion: FastAPI selected over Django. Need async support for real-time features."},
            {"id": 5, "text": "Deployment strategy: Going with Docker + Kubernetes on AWS. Auto-scaling is critical for launch."},
            {"id": 6, "text": "Budget review: Cloud costs estimated at $2000/month. Within budget allocation."},
            {"id": 7, "text": "Coffee chat about the new office layout. Open floor plan vs cubicles debate."},
            {"id": 8, "text": "Performance testing results: FastAPI handles 10K concurrent connections. Meets our requirements."},
            {"id": 9, "text": "Security audit: Need to add rate limiting and input validation before launch."},
        ],
        "evidence_ids": [0, 1, 4, 5],
        "facets": ["frontend_choice", "frontend_reason", "backend_choice", "backend_reason", "database_choice", "database_reason", "deployment_choice"],
        "facet_memory_map": {"frontend_choice": 0, "frontend_reason": 0, "backend_choice": 4, "backend_reason": 4, "database_choice": 1, "database_reason": 1, "deployment_choice": 5},
    },
    {
        "id": "temporal_1",
        "question": "How did the team's stance on remote work change over the three months?",
        "category": "temporal",
        "memories": [
            {"id": 0, "text": "January all-hands: CEO announced mandatory return to office 5 days a week starting February."},
            {"id": 1, "text": "Q4 revenue report showed 15% growth. Celebrated with team dinner."},
            {"id": 2, "text": "February survey results: 73% of employees reported decreased satisfaction with RTO policy."},
            {"id": 3, "text": "New coffee machine installed in the break room. Everyone loves it."},
            {"id": 4, "text": "February town hall: HR presented data showing 20% increase in turnover since RTO mandate."},
            {"id": 5, "text": "March policy update: CEO reversed course. Now hybrid 3 days in office, 2 remote. Cited retention data."},
            {"id": 6, "text": "IT upgraded the conference room AV equipment for better hybrid meetings."},
            {"id": 7, "text": "Quarterly OKR review. Team hit 4 of 5 objectives."},
            {"id": 8, "text": "March satisfaction survey: Employee satisfaction recovered to 85% after hybrid policy."},
            {"id": 9, "text": "Parking garage construction causing noise complaints from third floor."},
        ],
        "evidence_ids": [0, 2, 4, 5, 8],
        "facets": ["initial_policy", "employee_reaction", "turnover_impact", "policy_change", "final_outcome"],
        "facet_memory_map": {"initial_policy": 0, "employee_reaction": 2, "turnover_impact": 4, "policy_change": 5, "final_outcome": 8},
    },
    {
        "id": "causal_1",
        "question": "What caused the production outage, what was done to fix it, and what prevention measures were taken?",
        "category": "causal",
        "memories": [
            {"id": 0, "text": "Monday 2am alert: Production database hit 100% disk usage. All writes failing."},
            {"id": 1, "text": "Sprint retrospective: Team agreed to improve code review process."},
            {"id": 2, "text": "Root cause analysis: Logging table grew 500GB in 2 weeks due to debug logging left on after feature deploy."},
            {"id": 3, "text": "Incident response: DevOps team purged old log entries and increased disk from 1TB to 2TB."},
            {"id": 4, "text": "New hire orientation for three junior developers. HR handled logistics."},
            {"id": 5, "text": "Post-mortem action item 1: Implement log rotation with 30-day retention policy."},
            {"id": 6, "text": "Post-mortem action item 2: Add disk usage alerts at 70%, 80%, 90% thresholds."},
            {"id": 7, "text": "Post-mortem action item 3: Require removing debug logging before merging to main."},
            {"id": 8, "text": "Team building event at the escape room. Marketing team won."},
            {"id": 9, "text": "Client demo went well. They want to proceed with Phase 2."},
        ],
        "evidence_ids": [0, 2, 3, 5, 6, 7],
        "facets": ["what_happened", "root_cause", "immediate_fix", "prevention_1", "prevention_2", "prevention_3"],
        "facet_memory_map": {"what_happened": 0, "root_cause": 2, "immediate_fix": 3, "prevention_1": 5, "prevention_2": 6, "prevention_3": 7},
    },
    {
        "id": "multi_entity_1",
        "question": "What are each team member's roles and how do their responsibilities interact?",
        "category": "multi_entity",
        "memories": [
            {"id": 0, "text": "Alice leads frontend development. She works closely with Bob on API contracts."},
            {"id": 1, "text": "Company picnic was fun. Great weather this year."},
            {"id": 2, "text": "Bob owns the backend services. He designs APIs that Alice's frontend consumes."},
            {"id": 3, "text": "Carol manages the infrastructure. She provisions the servers Bob's services run on."},
            {"id": 4, "text": "New ping pong table in the break room. Tournament next Friday."},
            {"id": 5, "text": "Dave handles QA. He writes integration tests that cover Alice's UI and Bob's APIs."},
            {"id": 6, "text": "Eve is the project manager. She coordinates between Alice, Bob, Carol, and Dave."},
            {"id": 7, "text": "Office plants are dying. Need to assign someone to water them."},
            {"id": 8, "text": "Alice and Carol paired on improving the CI/CD pipeline. Reduced deploy time by 40%."},
            {"id": 9, "text": "Dave found a critical bug in Bob's API. Bob fixed it same day."},
        ],
        "evidence_ids": [0, 2, 3, 5, 6, 8, 9],
        "facets": ["alice_role", "bob_role", "carol_role", "dave_role", "eve_role", "alice_bob_interaction", "bob_carol_interaction", "dave_integration"],
        "facet_memory_map": {"alice_role": 0, "bob_role": 2, "carol_role": 3, "dave_role": 5, "eve_role": 6, "alice_bob_interaction": 0, "bob_carol_interaction": 3, "dave_integration": 5},
    },
    {
        "id": "synthesis_2",
        "question": "What is the complete customer onboarding process from signup to first value?",
        "category": "synthesis",
        "memories": [
            {"id": 0, "text": "Step 1: Customer signs up via website. Auto-creates account and sends welcome email."},
            {"id": 1, "text": "Marketing team redesigned the landing page. Conversion rate up 12%."},
            {"id": 2, "text": "Step 2: Customer success rep schedules onboarding call within 24 hours of signup."},
            {"id": 3, "text": "Step 3: During onboarding call, rep helps customer import their data and configure integrations."},
            {"id": 4, "text": "Sales team hit quarterly target. Pizza party celebration."},
            {"id": 5, "text": "Step 4: Customer gets access to interactive tutorial. Must complete 3 core modules."},
            {"id": 6, "text": "Step 5: After tutorial completion, customer success checks in at day 7 and day 30."},
            {"id": 7, "text": "Office AC broken again. Facilities contacted."},
            {"id": 8, "text": "Churn analysis: Customers who complete onboarding tutorial have 3x higher retention."},
            {"id": 9, "text": "Support ticket about login issues. Resolved — was a password reset problem."},
        ],
        "evidence_ids": [0, 2, 3, 5, 6],
        "facets": ["signup", "scheduling", "data_import", "tutorial", "followup"],
        "facet_memory_map": {"signup": 0, "scheduling": 2, "data_import": 3, "tutorial": 5, "followup": 6},
    },
]


def evaluate_combination(selected_ids: List[int], scenario: Dict) -> Dict:
    """
    Evaluate how well selected memories combine to answer the question.
    
    Returns facet coverage, synergy score, and overall combination quality.
    """
    evidence_ids = set(scenario["evidence_ids"])
    facet_map = scenario["facet_memory_map"]
    facets = scenario["facets"]
    selected_set = set(selected_ids)
    
    # Facet coverage: what percentage of required facets are covered?
    covered_facets = []
    for facet in facets:
        required_mem = facet_map[facet]
        if required_mem in selected_set:
            covered_facets.append(facet)
    
    coverage = len(covered_facets) / len(facets) if facets else 0
    
    # Evidence recall: what percentage of evidence memories selected?
    evidence_found = selected_set & evidence_ids
    evidence_recall = len(evidence_found) / len(evidence_ids) if evidence_ids else 0
    
    # Precision: what percentage of selected are actually evidence?
    precision = len(evidence_found) / len(selected_set) if selected_set else 0
    
    # Noise: non-evidence memories selected
    noise = len(selected_set - evidence_ids)
    
    return {
        "coverage": coverage,
        "evidence_recall": evidence_recall,
        "precision": precision,
        "noise": noise,
        "covered_facets": covered_facets,
        "missing_facets": [f for f in facets if f not in covered_facets],
        "f1": (2 * precision * evidence_recall / (precision + evidence_recall)
               if (precision + evidence_recall) > 0 else 0),
    }


def run_benchmark(recall_fn, K: int = 5, scenarios: List[Dict] = None) -> Dict:
    """
    Run MemCombine benchmark against a recall function.
    
    Args:
        recall_fn: Function(memories, query, K) -> List[int] (selected indices)
        K: Number of memories to select
        scenarios: Custom scenarios (uses built-in if None)
    
    Returns:
        Benchmark results with per-scenario and aggregate scores
    """
    if scenarios is None:
        scenarios = SCENARIOS
    
    results = []
    total_coverage = 0
    total_recall = 0
    total_f1 = 0
    perfect = 0
    
    for scenario in scenarios:
        memory_texts = [m["text"] for m in scenario["memories"]]
        selected = recall_fn(memory_texts, scenario["question"], K)
        
        eval_result = evaluate_combination(selected, scenario)
        
        results.append({
            "id": scenario["id"],
            "category": scenario["category"],
            "selected": selected,
            **eval_result,
        })
        
        total_coverage += eval_result["coverage"]
        total_recall += eval_result["evidence_recall"]
        total_f1 += eval_result["f1"]
        if eval_result["coverage"] == 1.0:
            perfect += 1
    
    n = len(scenarios)
    return {
        "benchmark": "MemCombine",
        "n_scenarios": n,
        "K": K,
        "avg_coverage": total_coverage / n,
        "avg_evidence_recall": total_recall / n,
        "avg_f1": total_f1 / n,
        "perfect_coverage": perfect,
        "perfect_coverage_pct": perfect / n * 100,
        "per_scenario": results,
    }
