"""
Obsidian Vault Exporter — Export QMG graph as linked markdown.

Converts the knowledge graph to an Obsidian vault where:
  - Each memory = one markdown note
  - Graph edges = [[wikilinks]] between notes
  - Entities = tags (#entity)
  - Agent = folder grouping

DK 🦍 — v1.0.0
"""

import os
import re
import json
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

from .graph import MemoryGraph, Memory


def sanitize_filename(text: str, max_len: int = 60) -> str:
    """Make text safe for filenames."""
    # Take first line or first N chars
    name = text.split('\n')[0][:max_len].strip()
    # Remove/replace unsafe chars
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    if not name:
        name = "untitled"
    return name


def memory_to_note(memory: Memory, graph: MemoryGraph,
                   agent_id: str = None) -> str:
    """Convert a Memory node to an Obsidian markdown note."""
    lines = []
    
    # Frontmatter
    lines.append("---")
    lines.append(f"id: {memory.id}")
    if memory.timestamp:
        lines.append(f"created: {memory.timestamp.isoformat()}")
    if memory.source:
        lines.append(f"source: {memory.source}")
    if agent_id:
        lines.append(f"agent: {agent_id}")
    if memory.entities:
        lines.append(f"entities: {json.dumps(memory.entities)}")
    lines.append("---")
    lines.append("")
    
    # Content
    lines.append(memory.text)
    lines.append("")
    
    # Connected memories as wikilinks
    neighbors = list(graph.graph.neighbors(memory.id)) if memory.id in graph.graph else []
    if neighbors:
        lines.append("## Connections")
        lines.append("")
        for nid in neighbors:
            neighbor = graph.memories.get(nid)
            if neighbor:
                edge_data = graph.graph[memory.id][nid]
                weight = edge_data.get('weight', 0)
                types = edge_data.get('types', [])
                
                link_name = sanitize_filename(neighbor.text)
                type_str = ", ".join(types) if types else "related"
                lines.append(f"- [[{link_name}]] ({type_str}, {weight:.2f})")
        lines.append("")
    
    # Entity tags
    if memory.entities:
        lines.append("## Tags")
        lines.append("")
        tags = " ".join(f"#{e.replace(' ', '_')}" for e in memory.entities)
        lines.append(tags)
        lines.append("")
    
    return "\n".join(lines)


def export_vault(graph: MemoryGraph, vault_path: str,
                 agent_id: str = None, agent_memories: Dict[str, List[str]] = None):
    """
    Export a MemoryGraph as an Obsidian vault.
    
    Args:
        graph: The memory graph to export
        vault_path: Path to create/update the Obsidian vault
        agent_id: If set, tag all notes with this agent
        agent_memories: Optional dict of {agent_id: [memory_ids]} for multi-agent vaults
    
    Creates:
        vault_path/
        ├── .obsidian/         (minimal config)
        ├── _index.md          (vault overview)
        ├── agent_name/        (per-agent folders if multi-agent)
        │   ├── memory1.md
        │   └── memory2.md
        └── shared/            (if shared pool exists)
    """
    vault = Path(vault_path).resolve()
    # Prevent path traversal — vault must be under user home or /tmp
    vault_str = str(vault)
    allowed_prefixes = [str(Path.home()), '/tmp', '/opt']
    if not any(vault_str.startswith(p) for p in allowed_prefixes):
        return {"ok": False, "error": f"Vault path must be under home, /tmp, or /opt"}
    vault.mkdir(parents=True, exist_ok=True)
    
    # Create minimal .obsidian config
    obsidian_dir = vault / ".obsidian"
    obsidian_dir.mkdir(exist_ok=True)
    
    # Enable graph view and backlinks
    app_config = {
        "showLineNumber": True,
        "strictLineBreaks": False,
        "useTab": False,
    }
    (obsidian_dir / "app.json").write_text(json.dumps(app_config, indent=2))
    
    # Graph view config — show all connections
    graph_config = {
        "collapse-filter": False,
        "search": "",
        "showTags": True,
        "showAttachments": False,
        "showOrphans": True,
        "collapse-color-groups": False,
        "colorGroups": [],
        "collapse-display": False,
        "showArrow": True,
        "textFadeMultiplier": 0,
        "nodeSizeMultiplier": 1,
        "lineSizeMultiplier": 1,
        "collapse-forces": False,
        "centerStrength": 0.518713248970312,
        "repelStrength": 10,
        "linkStrength": 1,
        "linkDistance": 250,
    }
    (obsidian_dir / "graph.json").write_text(json.dumps(graph_config, indent=2))
    
    # File tracking for wikilink consistency
    id_to_filename = {}
    exported_count = 0
    
    if agent_memories:
        # Multi-agent export — one folder per agent
        for aid, mem_ids in agent_memories.items():
            agent_dir = vault / aid
            agent_dir.mkdir(exist_ok=True)
            
            for mid in mem_ids:
                mem = graph.memories.get(mid)
                if not mem:
                    continue
                
                filename = sanitize_filename(mem.text)
                id_to_filename[mid] = f"{aid}/{filename}"
                
                note = memory_to_note(mem, graph, agent_id=aid)
                filepath = agent_dir / f"{filename}.md"
                filepath.write_text(note)
                exported_count += 1
    else:
        # Single agent or flat export
        folder = vault / agent_id if agent_id else vault
        folder.mkdir(exist_ok=True)
        
        for mid, mem in graph.memories.items():
            filename = sanitize_filename(mem.text)
            id_to_filename[mid] = filename
            
            note = memory_to_note(mem, graph, agent_id=agent_id)
            filepath = folder / f"{filename}.md"
            filepath.write_text(note)
            exported_count += 1
    
    # Create index note
    index = _build_index(graph, id_to_filename, agent_memories)
    (vault / "_index.md").write_text(index)
    
    return {
        "ok": True,
        "vault_path": str(vault),
        "notes_exported": exported_count,
        "graph_nodes": graph.graph.number_of_nodes(),
        "graph_edges": graph.graph.number_of_edges(),
    }


def _build_index(graph: MemoryGraph, id_to_filename: Dict,
                 agent_memories: Dict = None) -> str:
    """Build the vault index note."""
    lines = [
        "# 🧠 Quantum Memory Graph",
        "",
        f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Memories:** {graph.graph.number_of_nodes()}",
        f"**Connections:** {graph.graph.number_of_edges()}",
        "",
    ]
    
    stats = graph.stats()
    lines.append(f"**Density:** {stats['density']:.3f}")
    lines.append(f"**Components:** {stats['components']}")
    lines.append(f"**Avg connections per memory:** {stats['avg_degree']:.1f}")
    lines.append("")
    
    if agent_memories:
        lines.append("## Agents")
        lines.append("")
        for aid, mem_ids in sorted(agent_memories.items()):
            lines.append(f"- **{aid}**: {len(mem_ids)} memories → [[{aid}/]]")
        lines.append("")
    
    # Most connected memories
    if graph.memories:
        degrees = [(nid, graph.graph.degree(nid)) 
                    for nid in graph.graph.nodes() if nid in graph.memories]
        degrees.sort(key=lambda x: x[1], reverse=True)
        
        lines.append("## Most Connected Memories")
        lines.append("")
        for nid, deg in degrees[:10]:
            mem = graph.memories[nid]
            fname = id_to_filename.get(nid, sanitize_filename(mem.text))
            lines.append(f"- [[{fname}]] ({deg} connections)")
        lines.append("")
    
    return "\n".join(lines)


def export_from_mem0(mem0_url: str, vault_path: str,
                     api_token: str = "", agents: List[str] = None) -> Dict:
    """
    Export memories from a mem0 API into an Obsidian vault.
    
    Pulls memories from mem0, builds a QMG graph, then exports.
    This is the "one command" interface for the full pipeline.
    
    Usage:
        export_from_mem0(
            "http://localhost:8500",
            "/path/to/vault",
            agents=["donkeykong", "daisy", "luigi"]
        )
    """
    import requests
    from urllib.parse import urlparse
    
    # SSRF protection — only allow internal/known hosts
    parsed = urlparse(mem0_url)
    allowed_hosts = ['localhost', '127.0.0.1']
    # Add custom hosts via QMG_ALLOWED_HOSTS env var (comma-separated)
    extra = os.environ.get('QMG_ALLOWED_HOSTS', '')
    if extra:
        allowed_hosts.extend(h.strip() for h in extra.split(',') if h.strip())
    if parsed.hostname not in allowed_hosts and not parsed.hostname.endswith('.ts.net'):
        return {"ok": False, "error": f"mem0_url host not in allowlist: {parsed.hostname}"}
    
    all_agents = agents or [
        'bowser', 'mario', 'luigi', 'toadstool', 'yoshi', 'peach',
        'daisy', 'birdo', 'rosalina', 'koopa', 'wario', 'donkeykong'
    ]
    
    headers = {"Authorization": f"Bearer {api_token}"} if api_token else {}
    
    # Pull memories from mem0
    agent_memories_map = {}
    all_texts = []
    text_to_agent = {}
    
    for agent in all_agents:
        try:
            resp = requests.get(
                f"{mem0_url}/memories/{agent}",
                headers=headers, timeout=10
            )
            if resp.ok:
                data = resp.json()
                memories = data.get("memories", [])
                texts = [m.get("memory", m.get("text", "")) for m in memories if m.get("memory") or m.get("text")]
                agent_memories_map[agent] = texts
                for t in texts:
                    all_texts.append(t)
                    text_to_agent[t] = agent
        except Exception as e:
            print(f"  ⚠️ {agent}: {e}")
    
    if not all_texts:
        return {"ok": False, "error": "No memories found"}
    
    # Build QMG graph from all memories
    graph = MemoryGraph(similarity_threshold=0.3)
    graph.add_memories_batch(all_texts)
    
    # Map memory IDs to agents
    agent_mem_ids = {}
    for mid, mem in graph.memories.items():
        agent = text_to_agent.get(mem.text, "unknown")
        if agent not in agent_mem_ids:
            agent_mem_ids[agent] = []
        agent_mem_ids[agent].append(mid)
    
    # Export
    result = export_vault(graph, vault_path, agent_memories=agent_mem_ids)
    result["agents_exported"] = len(agent_mem_ids)
    result["total_memories_pulled"] = len(all_texts)
    return result
