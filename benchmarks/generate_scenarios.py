"""
Auto-generate MemCombine scenarios from real agent memories + synthetic expansion.
Generates 200+ multi-facet questions requiring combination of related memories.

DK 🦍
"""

import json
import random
import hashlib
from typing import List, Dict, Tuple
from itertools import combinations

# Scenario templates organized by category
TEMPLATES = {
    "synthesis": [
        {
            "pattern": "technology_stack",
            "question": "What is the complete {project} technology stack and why was each component chosen?",
            "facets": ["frontend", "backend", "database", "deployment", "reasoning"],
            "evidence_count": 4,
            "noise_ratio": 0.6,
        },
        {
            "pattern": "process_steps",
            "question": "What are the steps in the {process} workflow from start to finish?",
            "facets": ["step1", "step2", "step3", "step4", "verification"],
            "evidence_count": 5,
            "noise_ratio": 0.5,
        },
        {
            "pattern": "system_architecture",
            "question": "How is the {system} system architected and what are the key components?",
            "facets": ["component1", "component2", "component3", "integration", "config"],
            "evidence_count": 5,
            "noise_ratio": 0.5,
        },
        {
            "pattern": "feature_overview",
            "question": "What does the {feature} feature do and how was it built?",
            "facets": ["purpose", "implementation", "testing", "deployment"],
            "evidence_count": 4,
            "noise_ratio": 0.6,
        },
    ],
    "temporal": [
        {
            "pattern": "decision_evolution",
            "question": "How did the approach to {topic} change over time?",
            "facets": ["initial_state", "problem_identified", "change_made", "outcome"],
            "evidence_count": 4,
            "noise_ratio": 0.6,
        },
        {
            "pattern": "project_timeline",
            "question": "What was the timeline of the {project} project from inception to completion?",
            "facets": ["start", "development", "testing", "launch", "results"],
            "evidence_count": 5,
            "noise_ratio": 0.5,
        },
        {
            "pattern": "issue_resolution",
            "question": "How was the {issue} issue discovered, investigated, and resolved?",
            "facets": ["discovery", "investigation", "root_cause", "fix", "prevention"],
            "evidence_count": 5,
            "noise_ratio": 0.5,
        },
    ],
    "causal": [
        {
            "pattern": "incident_response",
            "question": "What caused the {incident} and what actions were taken?",
            "facets": ["trigger", "root_cause", "immediate_fix", "prevention_1", "prevention_2"],
            "evidence_count": 5,
            "noise_ratio": 0.5,
        },
        {
            "pattern": "decision_chain",
            "question": "What led to the decision about {decision} and what were the consequences?",
            "facets": ["context", "options", "decision", "implementation", "result"],
            "evidence_count": 5,
            "noise_ratio": 0.5,
        },
        {
            "pattern": "dependency_chain",
            "question": "What depends on {component} and what would break if it changed?",
            "facets": ["component_role", "dependent_1", "dependent_2", "config", "risk"],
            "evidence_count": 5,
            "noise_ratio": 0.5,
        },
    ],
    "multi_entity": [
        {
            "pattern": "team_roles",
            "question": "What are the roles and interactions between {entities}?",
            "facets": ["entity1_role", "entity2_role", "entity3_role", "interaction_1", "interaction_2"],
            "evidence_count": 5,
            "noise_ratio": 0.5,
        },
        {
            "pattern": "system_comparison",
            "question": "How do {system1} and {system2} compare in terms of features and performance?",
            "facets": ["system1_features", "system1_perf", "system2_features", "system2_perf", "comparison"],
            "evidence_count": 5,
            "noise_ratio": 0.5,
        },
        {
            "pattern": "cross_project",
            "question": "How do {project1} and {project2} relate to each other?",
            "facets": ["project1_purpose", "project2_purpose", "shared_components", "differences", "integration"],
            "evidence_count": 5,
            "noise_ratio": 0.5,
        },
    ],
}

# Domain knowledge pools for generating realistic content
DOMAINS = {
    "devops": {
        "projects": ["Chef OS", "Agent Platform", "Memory System", "API Gateway", "Dashboard"],
        "components": ["PostgreSQL", "Redis", "FastAPI", "React", "Docker", "Kubernetes", "Nginx", "Supabase"],
        "people": ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank"],
        "actions": ["deployed", "configured", "migrated", "optimized", "debugged", "refactored"],
        "noise_topics": ["office wifi", "lunch plans", "parking", "weather", "coffee machine",
                        "team building", "birthday cake", "desk arrangement", "plant watering",
                        "printer jam", "elevator music", "vending machine"],
    },
    "finance": {
        "projects": ["Budget Tracker", "Invoice System", "Payment Gateway", "Cost Analyzer", "Revenue Dashboard"],
        "components": ["Stripe", "QuickBooks", "Plaid", "ACH", "Wire Transfer", "Excel", "Tableau"],
        "people": ["CFO", "Controller", "Accountant", "Auditor", "Analyst", "Manager"],
        "actions": ["approved", "reconciled", "forecasted", "allocated", "audited", "invoiced"],
        "noise_topics": ["stock market", "cryptocurrency", "real estate", "insurance",
                        "company picnic", "holiday party", "new hire orientation"],
    },
    "restaurant": {
        "projects": ["Menu Planning", "Inventory System", "Order Management", "Recipe Database", "Catering Platform"],
        "components": ["Parsley", "Monday.com", "Otter", "Square POS", "ChefTec", "Supabase"],
        "people": ["Head Chef", "Sous Chef", "Line Cook", "Manager", "Server", "Bartender"],
        "actions": ["prepared", "plated", "ordered", "inventoried", "costed", "scheduled"],
        "noise_topics": ["table arrangement", "napkin folding", "music playlist", "bathroom soap",
                        "delivery truck schedule", "pest control", "flower arrangement"],
    },
    "security": {
        "projects": ["Auth System", "Firewall Config", "Audit Framework", "Token Manager", "Honeytoken Network"],
        "components": ["SSH", "fail2ban", "GPG", "JWT", "OAuth", "TLS", "WAF", "IDS"],
        "people": ["Security Lead", "DevOps", "Pen Tester", "Compliance Officer", "SysAdmin"],
        "actions": ["hardened", "patched", "monitored", "encrypted", "rotated", "audited"],
        "noise_topics": ["fire drill", "badge access", "visitor parking", "key fob battery",
                        "security camera angle", "door lock maintenance"],
    },
    "ai_ml": {
        "projects": ["Language Model", "Embedding Pipeline", "Fine-Tune System", "RAG Engine", "Agent Framework"],
        "components": ["PyTorch", "Transformers", "FAISS", "ChromaDB", "Ollama", "vLLM", "LoRA"],
        "people": ["ML Engineer", "Data Scientist", "Research Lead", "Infra Engineer", "Product Manager"],
        "actions": ["trained", "fine-tuned", "deployed", "benchmarked", "optimized", "evaluated"],
        "noise_topics": ["GPU availability", "cloud costs", "conference papers", "team lunch",
                        "whiteboard session", "standing desk", "monitor setup"],
    },
}


def generate_evidence_memory(domain: str, facet: str, project: str, idx: int) -> str:
    """Generate a realistic evidence memory for a specific facet."""
    d = DOMAINS[domain]
    comp = random.choice(d["components"])
    person = random.choice(d["people"])
    action = random.choice(d["actions"])
    
    templates = {
        "frontend": f"{person} {action} the frontend using {comp} for {project}. Chose it for its component ecosystem and TypeScript support.",
        "backend": f"Backend for {project} uses {comp}. {person} set it up with async request handling and automatic API documentation.",
        "database": f"{project} data stored in {comp}. {person} {action} the schema with proper indexing and RLS policies.",
        "deployment": f"Deployment pipeline for {project}: {comp} containers on auto-scaling infrastructure. {person} configured the CI/CD.",
        "reasoning": f"Architecture decision: {person} chose {comp} for {project} because of performance benchmarks and team familiarity.",
        "step1": f"Step 1 of {project}: {person} {action} the initial setup using {comp}. Created base configuration.",
        "step2": f"Step 2: {person} integrated {comp} into the {project} pipeline. Configured authentication and data flow.",
        "step3": f"Step 3: Testing phase for {project}. {person} wrote integration tests covering {comp} interactions.",
        "step4": f"Step 4: {person} {action} the staging deployment. {comp} passed all health checks.",
        "verification": f"Final verification: {person} confirmed {project} meets all requirements. {comp} performing within SLA.",
        "component1": f"{project} core: {comp} handles the main processing. {person} {action} it with custom middleware.",
        "component2": f"Supporting service: {comp} provides caching and session management for {project}.",
        "component3": f"Data layer: {comp} stores all persistent state for {project}. {person} optimized queries.",
        "integration": f"{person} connected {project} components: {comp} communicates via REST API with auth tokens.",
        "config": f"Configuration: {project} uses environment variables for {comp} settings. {person} documented the setup.",
        "purpose": f"{project} feature built to solve: automated {action} workflow. {person} identified the need.",
        "implementation": f"Implementation: {person} built {project} using {comp}. Core logic handles edge cases.",
        "testing": f"Testing: {person} wrote 15 tests for {project}. All passing. Covers {comp} integration.",
        "initial_state": f"Initially, {project} used a manual approach for {action}. {person} managed it with spreadsheets.",
        "problem_identified": f"Problem discovered: {project}'s manual {action} process was taking 4 hours daily. {person} flagged it.",
        "change_made": f"Changed {project} to automated {action} using {comp}. {person} led the migration.",
        "outcome": f"After change: {project} {action} time reduced from 4 hours to 5 minutes. {person} measured the improvement.",
        "start": f"{project} started when {person} proposed using {comp} for the {action} workflow.",
        "development": f"Development phase: {person} built {project} core in 2 weeks. {comp} integration took another week.",
        "launch": f"Launch: {project} went live. {person} monitored {comp} metrics during rollout.",
        "results": f"Results: {project} achieved 95% automation rate. {comp} handling 10K requests daily.",
        "discovery": f"Discovery: {person} noticed {project} {comp} errors spiking at 3am. Investigated immediately.",
        "investigation": f"Investigation: {person} traced {project} issue to {comp} configuration. Logs showed memory leak.",
        "root_cause": f"Root cause: {comp} in {project} had unbounded cache growth. {person} found the missing eviction policy.",
        "fix": f"Fix applied: {person} added cache limits to {comp} in {project}. Deployed hotfix.",
        "prevention": f"Prevention: {person} added monitoring alerts for {comp} memory usage in {project}.",
        "prevention_1": f"Prevention measure 1: {person} added automated alerts for {project} {comp} resource usage.",
        "prevention_2": f"Prevention measure 2: {person} implemented weekly {comp} health checks for {project}.",
        "trigger": f"Incident trigger: {project} {comp} went down due to disk full. {person} was paged.",
        "immediate_fix": f"Immediate fix: {person} cleared old logs and expanded {comp} storage for {project}.",
        "context": f"Context: {project} needed a better {action} solution. {person} evaluated 3 options including {comp}.",
        "options": f"Options evaluated for {project}: {comp}, two alternatives. {person} ran benchmarks on all three.",
        "decision": f"Decision: {person} chose {comp} for {project}. Best performance-to-cost ratio for our scale.",
        "result": f"Result: {project} with {comp} handles 5x more load than the previous solution. {person} validated.",
        "component_role": f"{comp} serves as the {action} layer in {project}. {person} maintains it.",
        "dependent_1": f"The {project} API depends on {comp} for data access. {person} manages the connection pool.",
        "dependent_2": f"Monitoring dashboard reads from {comp} in {project}. {person} built the integration.",
        "risk": f"Risk: if {comp} goes down, {project} loses {action} capability. {person} set up failover.",
        "entity1_role": f"{person} leads frontend development on {project}. Works with {comp} daily.",
        "entity2_role": f"{random.choice(d['people'])} handles backend services for {project}. Manages {comp} deployments.",
        "entity3_role": f"{random.choice(d['people'])} manages infrastructure for {project}. Provisions {comp} resources.",
        "interaction_1": f"{person} collaborates with the backend team on {project} API contracts. Uses {comp} for testing.",
        "interaction_2": f"Cross-team sync: {person} reviews {comp} changes in {project} with the infra team weekly.",
        "system1_features": f"{project} features: {action} automation, {comp} integration, real-time monitoring.",
        "system1_perf": f"{project} performance: handles 10K requests/sec with {comp}. 99.9% uptime.",
        "system2_features": f"Alternative system features: manual {action}, basic {comp} support, batch processing.",
        "system2_perf": f"Alternative performance: handles 1K requests/sec. 99% uptime. Higher latency.",
        "comparison": f"Comparison: {project} with {comp} is 10x faster than the alternative. {person} benchmarked both.",
        "project1_purpose": f"{project} purpose: automate {action} workflow using {comp}.",
        "project2_purpose": f"Related project purpose: provide {comp} data for {project}'s {action} pipeline.",
        "shared_components": f"Shared: both projects use {comp} for data storage. {person} manages the shared instance.",
        "differences": f"Difference: {project} is real-time, the other is batch. Different {comp} access patterns.",
    }
    
    return templates.get(facet, f"{person} {action} {comp} for {project}. Related to {facet}.")


def generate_noise_memory(domain: str) -> str:
    """Generate a realistic but irrelevant memory."""
    d = DOMAINS[domain]
    topic = random.choice(d["noise_topics"])
    person = random.choice(d["people"])
    
    noise_templates = [
        f"{person} mentioned the {topic} situation during standup.",
        f"Quick note about {topic}: {person} will handle it by Friday.",
        f"Team discussed {topic}. {person} volunteered to take care of it.",
        f"Reminder from {person}: {topic} needs attention this week.",
        f"{topic} update from {person}: everything is sorted now.",
        f"Side conversation about {topic}. {person} had a funny story.",
        f"{person} brought up {topic} after the meeting. Not urgent.",
        f"FYI: {person} said the {topic} is being addressed.",
        f"Meeting went off-topic: {person} talked about {topic} for 10 minutes.",
        f"{person} sent a message about {topic}. Low priority.",
    ]
    return random.choice(noise_templates)


def generate_scenario(template: Dict, domain: str, scenario_id: str) -> Dict:
    """Generate a complete MemCombine scenario from a template."""
    d = DOMAINS[domain]
    project = random.choice(d["projects"])
    
    # Format question
    question = template["question"].format(
        project=project, process=project, system=project,
        feature=project, topic=project.lower(), issue=project.lower(),
        incident=project.lower(), decision=project.lower(),
        component=random.choice(d["components"]),
        entities=", ".join(random.sample(d["people"], min(3, len(d["people"])))),
        system1=project, system2=random.choice(d["projects"]),
        project1=project, project2=random.choice(d["projects"]),
    )
    
    facets = template["facets"]
    evidence_count = template["evidence_count"]
    noise_ratio = template["noise_ratio"]
    
    # Generate evidence memories
    total_memories = int(evidence_count / (1 - noise_ratio))
    total_memories = max(total_memories, 8)
    total_memories = min(total_memories, 12)
    noise_count = total_memories - evidence_count
    
    # Create evidence memories
    evidence_memories = []
    for i, facet in enumerate(facets[:evidence_count]):
        text = generate_evidence_memory(domain, facet, project, i)
        evidence_memories.append({"id": i, "text": text, "is_evidence": True, "facet": facet})
    
    # Create noise memories
    noise_memories = []
    for i in range(noise_count):
        text = generate_noise_memory(domain)
        noise_memories.append({"id": evidence_count + i, "text": text, "is_evidence": False, "facet": None})
    
    # Shuffle
    all_memories = evidence_memories + noise_memories
    random.shuffle(all_memories)
    
    # Re-index
    id_map = {}
    for new_idx, mem in enumerate(all_memories):
        id_map[mem["id"]] = new_idx
        mem["id"] = new_idx
    
    evidence_ids = [mem["id"] for mem in all_memories if mem["is_evidence"]]
    facet_map = {}
    for mem in all_memories:
        if mem["facet"]:
            facet_map[mem["facet"]] = mem["id"]
    
    return {
        "id": scenario_id,
        "question": question,
        "category": template["pattern"],
        "domain": domain,
        "memories": [{"id": m["id"], "text": m["text"]} for m in all_memories],
        "evidence_ids": evidence_ids,
        "facets": facets,
        "facet_memory_map": facet_map,
    }


def generate_all_scenarios(count: int = 250, seed: int = 42) -> List[Dict]:
    """Generate a full benchmark suite."""
    random.seed(seed)
    scenarios = []
    
    domains = list(DOMAINS.keys())
    categories = list(TEMPLATES.keys())
    
    idx = 0
    while len(scenarios) < count:
        for category in categories:
            for template in TEMPLATES[category]:
                for domain in domains:
                    if len(scenarios) >= count:
                        break
                    scenario_id = f"{category}_{template['pattern']}_{domain}_{idx}"
                    scenario = generate_scenario(template, domain, scenario_id)
                    scenarios.append(scenario)
                    idx += 1
    
    random.shuffle(scenarios)
    return scenarios[:count]


if __name__ == "__main__":
    scenarios = generate_all_scenarios(250)
    
    output_path = "/home/dt/Projects/quantum-memory-graph/benchmarks/memcombine_250.json"
    with open(output_path, "w") as f:
        json.dump(scenarios, f, indent=2)
    
    # Stats
    cats = {}
    doms = {}
    for s in scenarios:
        cats[s["category"]] = cats.get(s["category"], 0) + 1
        doms[s["domain"]] = doms.get(s["domain"], 0) + 1
    
    print(f"Generated {len(scenarios)} scenarios")
    print(f"Categories: {cats}")
    print(f"Domains: {doms}")
    print(f"Avg memories/scenario: {sum(len(s['memories']) for s in scenarios)/len(scenarios):.1f}")
    print(f"Avg evidence/scenario: {sum(len(s['evidence_ids']) for s in scenarios)/len(scenarios):.1f}")
    print(f"Saved to {output_path}")
