"""
QMG Benchmark Data Collector.

Logs every benchmark run with full metadata for study and analysis.
Data is append-only, timestamped, and structured for easy analysis.

DK 🦍

Usage:
    from benchmarks.data_collector import QMGBenchmarkLogger
    
    logger = QMGBenchmarkLogger()
    logger.log_memcombine_run(method, results, params)
    logger.log_qaoa_run(n_candidates, n_qubits, selection_result, timing)
    logger.export_csv()  # Export all data for study
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path


BENCHMARK_LOG_DIR = Path(os.environ.get(
    "QMG_BENCHMARK_LOG",
    Path.home() / ".local" / "share" / "qmg" / "benchmarks",
))


class QMGBenchmarkLogger:
    """Append-only benchmark data logger."""
    
    def __init__(self, log_dir: str = None):
        self.log_dir = Path(log_dir) if log_dir else BENCHMARK_LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def log_memcombine_run(self, method: str, results: Dict, params: Dict = None):
        """Log a MemCombine benchmark run."""
        entry = {
            "type": "memcombine_run",
            "session_id": self._session_id,
            "timestamp": datetime.now().isoformat(),
            "unix_time": time.time(),
            "method": method,
            "results": {
                "coverage": results.get("avg_coverage", results.get("coverage")),
                "evidence_recall": results.get("avg_evidence_recall", results.get("evidence_recall")),
                "f1": results.get("avg_f1", results.get("f1")),
                "perfect": results.get("perfect_coverage", results.get("perfect")),
                "perfect_pct": results.get("perfect_coverage_pct", results.get("perfect_pct")),
                "n_scenarios": results.get("n_scenarios", len(results.get("per_scenario", []))),
            },
            "params": params or {},
            "per_scenario": results.get("per_scenario", []),
        }
        self._write("memcombine", entry)
        return entry
    
    def log_longmemeval_run(self, method: str, results: Dict, model: str = None, params: Dict = None):
        """Log a LongMemEval benchmark run."""
        entry = {
            "type": "longmemeval_run",
            "session_id": self._session_id,
            "timestamp": datetime.now().isoformat(),
            "unix_time": time.time(),
            "model": model or "unknown",
            "method": method,
            "results": {
                "recall_at_5": results.get("recall_at_5"),
                "recall_at_10": results.get("recall_at_10"),
                "ndcg_at_10": results.get("ndcg_at_10"),
                "n": results.get("n"),
            },
            "params": params or {},
        }
        self._write("longmemeval", entry)
        return entry
    
    def log_qaoa_run(self, 
                      n_candidates: int, 
                      n_qubits: int, 
                      K: int,
                      p_layers: int,
                      method: str,
                      selection_result: Dict,
                      timing_ms: float,
                      params: Dict = None):
        """Log a single QAOA optimization run."""
        entry = {
            "type": "qaoa_run",
            "session_id": self._session_id,
            "timestamp": datetime.now().isoformat(),
            "unix_time": time.time(),
            "n_candidates": n_candidates,
            "n_qubits": n_qubits,
            "K": K,
            "p_layers": p_layers,
            "method": method,
            "compression_ratio": selection_result.get("compression_ratio", f"{n_candidates}→{n_qubits}"),
            "qaoa_score": selection_result.get("score"),
            "qaoa_vs_greedy_pct": selection_result.get("qaoa_vs_greedy_pct"),
            "qaoa_vs_optimal_pct": selection_result.get("qaoa_vs_optimal_pct"),
            "timing_ms": timing_ms,
            "params": params or {},
        }
        self._write("qaoa", entry)
        return entry
    
    def log_graph_stats(self, graph_stats: Dict, tag: str = ""):
        """Log memory graph statistics."""
        entry = {
            "type": "graph_stats",
            "session_id": self._session_id,
            "timestamp": datetime.now().isoformat(),
            "unix_time": time.time(),
            "tag": tag,
            "nodes": graph_stats.get("nodes"),
            "edges": graph_stats.get("edges"),
            "density": graph_stats.get("density"),
            "components": graph_stats.get("components"),
            "avg_degree": graph_stats.get("avg_degree"),
        }
        self._write("graph", entry)
        return entry
    
    def log_hardware_run(self, backend: str, n_qubits: int, result: Dict, timing_ms: float):
        """Log a hardware execution run (IBM Quantum)."""
        entry = {
            "type": "hardware_run",
            "session_id": self._session_id,
            "timestamp": datetime.now().isoformat(),
            "unix_time": time.time(),
            "backend": backend,
            "n_qubits": n_qubits,
            "result": {
                "score": result.get("score"),
                "method": result.get("method"),
                "error_mitigation": result.get("error_mitigation", "none"),
            },
            "timing_ms": timing_ms,
        }
        self._write("hardware", entry)
        return entry
    
    def _write(self, category: str, entry: Dict):
        """Append entry to category log file."""
        log_file = self.log_dir / f"{category}_log.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
    
    def export_csv(self, category: str = None) -> str:
        """
        Export logged data as CSV for analysis.
        Returns path to CSV file.
        """
        import csv
        from collections import OrderedDict
        
        categories = [category] if category else ["memcombine", "longmemeval", "qaoa", "graph", "hardware"]
        output_paths = []
        
        for cat in categories:
            log_file = self.log_dir / f"{cat}_log.jsonl"
            if not log_file.exists():
                continue
            
            csv_path = self.log_dir / f"{cat}_export.csv"
            entries = []
            with open(log_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
            
            if not entries:
                continue
            
            # Gather all keys
            all_keys = []
            for e in entries:
                for k in e.keys():
                    if k not in all_keys:
                        all_keys.append(k)
            
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
                writer.writeheader()
                for e in entries:
                    writer.writerow(e)
            
            output_paths.append(str(csv_path))
        
        return ", ".join(output_paths) if output_paths else "No data logged yet"
    
    def summary(self) -> Dict:
        """Quick summary of all logged data."""
        result = {}
        for cat in ["memcombine", "longmemeval", "qaoa", "graph", "hardware"]:
            log_file = self.log_dir / f"{cat}_log.jsonl"
            if log_file.exists():
                with open(log_file) as f:
                    lines = [l for l in f if l.strip()]
                result[cat] = len(lines)
            else:
                result[cat] = 0
        return result
