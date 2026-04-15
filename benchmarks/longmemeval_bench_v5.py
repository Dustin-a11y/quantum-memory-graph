"""
LongMemEval v5 — Turn-level chunking.
Instead of dumb 500-char slices, chunk at natural dialog turn boundaries.
Each chunk = 1-3 complete user/assistant exchanges. Facts stay intact.

DK 🦍
"""

import json, sys, os, time, math, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Loading model...", flush=True)
from sentence_transformers import SentenceTransformer
MODEL = SentenceTransformer('all-MiniLM-L6-v2')
print("Ready.", flush=True)


def session_to_turns(session):
    """Convert a session into individual dialog turns (user+assistant pairs)."""
    if isinstance(session, str):
        # Plain text — split on newlines, group every 2-3 lines
        lines = [l.strip() for l in session.split('\n') if l.strip()]
        turns = []
        for i in range(0, len(lines), 2):
            turn = '\n'.join(lines[i:i+2])
            if len(turn) > 30:
                turns.append(turn)
        return turns if turns else [session]
    
    if isinstance(session, list):
        turns = []
        current_turn = []
        for item in session:
            if isinstance(item, dict):
                role = item.get("role", "")
                content = item.get("content", item.get("text", str(item)))
                current_turn.append(f"{role}: {content}" if role else content)
                # Group by user+assistant pair
                if role in ("assistant", "bot", "system") and len(current_turn) >= 2:
                    turns.append('\n'.join(current_turn))
                    current_turn = []
            else:
                current_turn.append(str(item))
        if current_turn:
            turns.append('\n'.join(current_turn))
        return turns if turns else [str(session)]
    
    return [str(session)]


def chunk_turns(session, max_turns_per_chunk=3):
    """Chunk at turn boundaries. Each chunk = 1-3 complete exchanges."""
    turns = session_to_turns(session)
    chunks = []
    for i in range(0, len(turns), max_turns_per_chunk):
        chunk = '\n'.join(turns[i:i+max_turns_per_chunk])
        if len(chunk) > 30:
            chunks.append(chunk)
    # Also add overlapping chunks (shifted by 1 turn) for coverage
    if len(turns) > max_turns_per_chunk:
        for i in range(1, len(turns), max_turns_per_chunk):
            chunk = '\n'.join(turns[i:i+max_turns_per_chunk])
            if len(chunk) > 30:
                chunks.append(chunk)
    return chunks if chunks else [str(session) if not isinstance(session, str) else session]


def flatten_session(session):
    if isinstance(session, str):
        return session
    if isinstance(session, list):
        parts = []
        for turn in session:
            if isinstance(turn, dict):
                role = turn.get("role", "")
                content = turn.get("content", turn.get("text", str(turn)))
                parts.append(f"{role}: {content}" if role else content)
            else:
                parts.append(str(turn))
        return "\n".join(parts)
    return str(session)


def recall_at_k(retrieved, gold, K):
    return len(set(retrieved[:K]) & set(gold)) / len(gold) if gold else 1.0


def ndcg_at_k(retrieved, gold, K):
    gold_set = set(gold)
    if not gold_set:
        return 1.0
    dcg = sum(1.0 / math.log2(i + 2) for i, idx in enumerate(retrieved[:K]) if idx in gold_set)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold_set), K)))
    return dcg / idcg if idcg > 0 else 0.0


def topk_session_scores(chunk_embs, chunk_to_session, q_emb):
    scores = np.dot(chunk_embs, q_emb)
    # Best chunk score per session
    session_best = {}
    session_chunk_scores = {}
    for ci, score in enumerate(scores):
        si = chunk_to_session[ci]
        if si not in session_best or score > session_best[si]:
            session_best[si] = float(score)
        if si not in session_chunk_scores:
            session_chunk_scores[si] = []
        session_chunk_scores[si].append(float(score))
    # Robust: mean of top-3 chunks
    session_robust = {}
    for si, cs in session_chunk_scores.items():
        top3 = sorted(cs, reverse=True)[:3]
        session_robust[si] = np.mean(top3)
    return session_best, session_robust


def topk_only(chunk_embs, chunk_to_session, q_emb, K=5):
    _, session_robust = topk_session_scores(chunk_embs, chunk_to_session, q_emb)
    ranked = sorted(session_robust.keys(), key=lambda s: session_robust[s], reverse=True)
    return ranked[:K*2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--turns", type=int, default=3, help="Max turns per chunk")
    args = parser.parse_args()

    print("Loading data...", flush=True)
    with open(args.data) as f:
        data = json.load(f)
    if args.limit:
        data = data[:args.limit]
    print(f"Running {len(data)} questions\n", flush=True)

    # Compare: char chunking vs turn chunking
    for chunk_mode in ["char", "turn1", "turn2", "turn3"]:
        print(f"\n{'='*60}", flush=True)
        print(f"Chunking mode: {chunk_mode}", flush=True)
        print(f"{'='*60}", flush=True)

        t0 = time.time()
        question_data = []
        all_chunks_flat = []

        for qi, item in enumerate(data):
            sessions = item["haystack_sessions"]
            q_chunks = []
            q_chunk_to_session = []

            for si, sess in enumerate(sessions):
                if chunk_mode == "char":
                    # Original 500-char chunking
                    text = flatten_session(sess)
                    start = 0
                    while start < len(text):
                        end = start + 500
                        chunk = text[start:end]
                        if len(chunk.strip()) > 50:
                            q_chunks.append(chunk)
                            q_chunk_to_session.append(si)
                            all_chunks_flat.append(chunk)
                        start = end - 100
                elif chunk_mode.startswith("turn"):
                    max_turns = int(chunk_mode[4:])
                    chunks = chunk_turns(sess, max_turns_per_chunk=max_turns)
                    for c in chunks:
                        q_chunks.append(c)
                        q_chunk_to_session.append(si)
                        all_chunks_flat.append(c)

            question_data.append({
                "chunks": q_chunks,
                "chunk_to_session": q_chunk_to_session,
                "gold_ids": item["answer_session_ids"],
                "all_ids": item["haystack_session_ids"],
                "question": item["question"],
            })

        print(f"  Total chunks: {len(all_chunks_flat)}", flush=True)
        print(f"  Avg chunks/question: {len(all_chunks_flat)/len(data):.0f}", flush=True)
        print(f"  Embedding...", flush=True)
        all_chunk_embs = MODEL.encode(all_chunks_flat, normalize_embeddings=True, batch_size=256, show_progress_bar=True)
        q_texts = [item["question"] for item in data]
        q_embs = MODEL.encode(q_texts, normalize_embeddings=True, batch_size=256)

        idx = 0
        for qi, qd in enumerate(question_data):
            n = len(qd["chunks"])
            qd["chunk_embs"] = all_chunk_embs[idx:idx+n]
            idx += n

        embed_time = time.time() - t0
        print(f"  Embed time: {embed_time:.0f}s", flush=True)

        # Run Top-K
        t0 = time.time()
        r5 = r10 = n10 = 0
        for qi, qd in enumerate(question_data):
            gold = [qd["all_ids"].index(g) for g in qd["gold_ids"] if g in qd["all_ids"]]
            retrieved = topk_only(qd["chunk_embs"], qd["chunk_to_session"], q_embs[qi])
            r5 += recall_at_k(retrieved, gold, 5)
            r10 += recall_at_k(retrieved, gold, 10)
            n10 += ndcg_at_k(retrieved, gold, 10)
            if (qi+1) % 100 == 0:
                n = qi+1
                print(f"  [{n}/{len(data)}] R@5={r5/n:.3f} R@10={r10/n:.3f} NDCG={n10/n:.3f}", flush=True)

        n = len(data)
        print(f"\n  FINAL [{chunk_mode}]: R@5={r5/n:.3f}  R@10={r10/n:.3f}  NDCG@10={n10/n:.3f}  ({time.time()-t0:.0f}s)", flush=True)

        # Reset for next mode
        all_chunks_flat = []

    # Final comparison table
    print(f"\n{'='*60}", flush=True)
    print(f"Previous SOTA:  R@5=0.966  R@10=0.982  NDCG=0.889", flush=True)
    print(f"Previous best (v4 char):  R@5=0.934  R@10=0.974  NDCG=0.908", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
