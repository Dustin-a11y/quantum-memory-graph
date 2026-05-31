#!/usr/bin/env python3
"""
LongMemEval End-to-End QA v2 — R1 reasoning model + more context + better prompt.

Changes from v1:
- Gen model: deepseek-reasoner (R1) instead of deepseek-v4-pro
- Top-K: 10 instead of 5 (more context for multi-session)
- Improved gen prompt: explicit aggregation + temporal reasoning instructions
- Session metadata (date info) included in context
- Less conservative abstention behavior

DK 🦍 — May 29, 2026
"""
import json, os, sys, time, re
from datetime import datetime, timezone
import numpy as np
import requests

DATA_PATH = "/home/dt/projects-shared/LongMemEval/data/longmemeval_s_cleaned.json"
OUTPUT_DIR = "/home/dt/qmg-v1/benchmarks"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "longmemeval_e2e_r1_results.json")
STATUS_FILE = "/home/dt/r1_status.txt"  # heartbeat file
TOP_K = 10  # increased from 5
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

DEEPSEEK_KEY = "sk-8804f6529df549dda014ae1b26baea24"
DEEPSEEK_URL = "https://api.deepseek.com/v1"
GEN_MODEL = "deepseek-reasoner"     # R1 reasoning model
JUDGE_MODEL = "deepseek-chat"       # verified working for judge

# ---- Helpers ----

def llm_call(messages, model=GEN_MODEL, max_tokens=1024, temperature=0, timeout=60):
    """Call DeepSeek API with retry"""
    last_e = None
    for attempt in range(3):
        try:
            resp = requests.post(
                f"{DEEPSEEK_URL}/chat/completions",
                headers={"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"},
                json={"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature},
                timeout=timeout
            )
            j = resp.json()
            content = j["choices"][0]["message"]["content"].strip() if j["choices"][0]["message"].get("content", "").strip() else ""
            
            # For reasoning model, also extract reasoning if available
            reasoning = j["choices"][0]["message"].get("reasoning_content", "")
            
            if content:
                return content
            if reasoning and not content:
                # Some R1 responses only have reasoning, no content
                return f"[Only reasoning provided: {reasoning[:200]}]"
            # Empty — retry
            print(f"  [WARN] Empty content attempt {attempt+1}", flush=True)
        except Exception as e:
            last_e = e
            print(f"  [WARN] Attempt {attempt+1} failed: {e}", flush=True)
            time.sleep(2 ** attempt)
    return f"[GEN FAILED: {last_e}]" if last_e else "[GEN FAILED: empty content]"

def flatten_session(session):
    if isinstance(session, str): return session
    if isinstance(session, list):
        parts = []
        for turn in session:
            if isinstance(turn, dict):
                parts.append(f"{turn.get('role','')}: {turn.get('content', turn.get('text', str(turn)))}")
            else:
                parts.append(str(turn))
        return "\n".join(parts)
    return str(session)

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
        start = end - overlap
    return chunks if chunks else [text]

def extract_timestamp(session_data):
    """Try to extract timestamp info from session data for temporal context"""
    if isinstance(session_data, dict):
        for key in ['timestamp', 'date', 'created_at', 'time']:
            if key in session_data:
                return str(session_data[key])
    return ""

def build_gen_prompt(question, sessions_data, top_k=TOP_K):
    """Build prompt with session context, temporal info, and explicit aggregation instructions"""
    
    sessions_text_parts = []
    for i, (text, ts) in enumerate(sessions_data):
        ts_str = f" [Timestamp: {ts}]" if ts else ""
        sessions_text_parts.append(f"Session {i+1}:{ts_str}\n{text}")
    
    full_context = "\n\n".join(sessions_text_parts)
    
    system_msg = (
        "You are a precise analyst. Answer the user's question using ONLY the provided conversation sessions. "
        "Be precise and concise — a short answer is best."
    )
    
    user_msg = (
        f"I have provided {len(sessions_data)} conversation sessions below. "
        f"These sessions may contain relevant information for answering the question.\n\n"
        f"IMPORTANT INSTRUCTIONS:\n"
        f"1. Use ALL provided sessions — the answer may span multiple sessions\n"
        f"2. If sessions have timestamps, use the LATEST information for questions about current state\n"
        f"3. For counting questions (\"how many X\"), count across ALL sessions\n"
        f"4. For temporal questions (\"when\", \"how long\", \"in what order\"), reason about dates\n"
        f"5. If the answer is definitively not in any session, say \"I cannot answer\"\n"
        f"6. Otherwise, do your best to synthesize the answer from the available information\n\n"
        f"--- SESSIONS ---\n\n{full_context}\n\n"
        f"--- QUESTION ---\n\n{question}\n\nANSWER:"
    )
    
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

def get_judge_prompt(qtype, question, answer, response):
    abstention = "_abs" in str(question)
    if abstention:
        return [{"role": "user", "content": (
            "I will give you an unanswerable question, an explanation, and a response from a model. "
            "Please answer yes if the model correctly identifies the question as unanswerable. "
            "The model could say that the information is incomplete, or some other information is given "
            "but the asked information is not.\n\n"
            f"Question: {question}\n\nExplanation: {answer}\n\nModel Response: {response}\n\n"
            "Does the model correctly identify the question as unanswerable? Answer yes or no only."
        )}]

    templates = {
        'single-session-user': (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response is equivalent to the correct answer or contains all the intermediate steps "
            "to get the correct answer, you should also answer yes. If the response only contains a subset "
            "of the information required by the answer, answer no.\n\n"
            "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
            "Is the model response correct? Answer yes or no only."
        ),
        'single-session-assistant': (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response is equivalent to the correct answer or contains all the intermediate steps "
            "to get the correct answer, you should also answer yes. If the response only contains a subset "
            "of the information required by the answer, answer no.\n\n"
            "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
            "Is the model response correct? Answer yes or no only."
        ),
        'multi-session': (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response is equivalent to the correct answer or contains all the intermediate steps "
            "to get the correct answer, you should also answer yes. If the response only contains a subset "
            "of the information required by the answer, answer no.\n\n"
            "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
            "Is the model response correct? Answer yes or no only."
        ),
        'temporal-reasoning': (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response is equivalent to the correct answer or contains all the intermediate steps "
            "to get the correct answer, you should also answer yes. If the response only contains a subset "
            "of the information required by the answer, answer no. In addition, do not penalize off-by-one "
            "errors for the number of days. If the question asks for the number of days/weeks/months, etc., "
            "and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), "
            "the model's response is still correct.\n\n"
            "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
            "Is the model response correct? Answer yes or no only."
        ),
        'knowledge-update': (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response contains some previous information along with an updated answer, "
            "the response should be considered as correct as long as the updated answer is the required answer.\n\n"
            "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
            "Is the model response correct? Answer yes or no only."
        ),
        'single-session-preference': (
            "I will give you a question, a rubric for desired personalized response, and a response from a model. "
            "Please answer yes if the response satisfies the desired response. Otherwise, answer no. "
            "The model does not need to reflect all the points in the rubric. The response is correct as long "
            "as it recalls and utilizes the user's personal information correctly.\n\n"
            "Question: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
            "Is the model response correct? Answer yes or no only."
        ),
    }
    t = templates.get(qtype, templates['single-session-user'])
    return [{"role": "user", "content": t.format(question, answer, response)}]

# ---- Main ----

def main():
    T_START = time.time()
    print("=" * 70)
    print(f"LongMemEval E2E QA v2 — GEN={GEN_MODEL} JUDGE={JUDGE_MODEL} top-{TOP_K}")
    print("=" * 70, flush=True)
    
    print("Loading LongMemEval data...", flush=True)
    with open(DATA_PATH) as f: ref_data = json.load(f)

    print(f"Loaded {len(ref_data)} questions. Loading gte-large...", flush=True)
    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("thenlper/gte-large", device=device)
    print(f"Model on {device}. Pre-computing all session embeddings...", flush=True)

    # Build data structures
    all_haystack_texts = {}
    all_haystack_sessions = {}
    all_questions = []
    
    from collections import OrderedDict
    unique_sessions = OrderedDict()
    for item in ref_data:
        qid = item["question_id"]
        haystack = item.get("haystack_sessions", [])
        texts = [flatten_session(s) for s in haystack]
        all_haystack_texts[qid] = texts
        all_haystack_sessions[qid] = haystack
        all_questions.append((qid, item["question"], item.get("question_type", "unknown"), item.get("answer", "")))
        
        for si, text in enumerate(texts):
            if text not in unique_sessions:
                unique_sessions[text] = []
            unique_sessions[text].append((qid, si))
    
    print(f"Total unique sessions across all questions: {len(unique_sessions)}", flush=True)
    
    # Chunk and encode all unique sessions
    print("Chunking and encoding all unique sessions...", flush=True)
    all_chunks = []
    chunk_metadata = []  # (session_text, chunk_idx)
    t0 = time.time()
    
    for sess_text in unique_sessions:
        chunks = chunk_text(sess_text)
        for ci, c in enumerate(chunks):
            all_chunks.append(c)
            chunk_metadata.append((sess_text, ci))
    
    print(f"Total chunks: {len(all_chunks)}", flush=True)
    
    if all_chunks:
        # Batch encode all chunks at once (big batch but single GPU pass)
        print(f"Encoding {len(all_chunks)} chunks on {device}...", flush=True)
        all_embeddings = model.encode(all_chunks, normalize_embeddings=True, batch_size=256, show_progress_bar=True)
        print(f"Encoding done in {time.time()-t0:.0f}s", flush=True)
    else:
        all_embeddings = np.array([])
    
    # Build per-session embeddings (mean of top-3 chunks)
    session_embeddings = {}
    current_idx = 0
    for sess_text in unique_sessions:
        n_chunks = len(chunk_text(sess_text))
        if n_chunks > 0:
            chunk_embs = all_embeddings[current_idx:current_idx + n_chunks]
            # Mean of top-3
            scores = np.sort(np.linalg.norm(chunk_embs, axis=1))[-3:] if n_chunks >= 3 else np.linalg.norm(chunk_embs, axis=1)
            session_embeddings[sess_text] = np.mean(chunk_embs[scores.argsort()[-min(3, n_chunks):]], axis=0)
            current_idx += n_chunks
        else:
            session_embeddings[sess_text] = np.zeros(1024)
    
    print(f"Session embeddings computed in {time.time()-t0:.0f}s", flush=True)
    
    # Now rank per question using pre-computed embeddings
    print("Computing retrieval scores...", flush=True)
    question_rankings = {}
    
    for idx, (qid, question, qtype, answer_key) in enumerate(all_questions):
        haystack = all_haystack_sessions[qid]
        if not haystack or len(haystack) < 3:
            question_rankings[qid] = None
            continue
        
        # Query encode (single query, fast)
        q_emb = model.encode([question], normalize_embeddings=True, show_progress_bar=False)[0]
        
        # Score all sessions using pre-computed embeddings
        session_scores = {}
        for si, text in enumerate(all_haystack_texts[qid]):
            if text in session_embeddings:
                session_scores[si] = float(session_embeddings[text] @ q_emb)
            else:
                session_scores[si] = -1.0
        
        ranking = sorted(range(len(haystack)), key=lambda i: session_scores.get(i, -1.0), reverse=True)
        question_rankings[qid] = ranking
        
        with open(STATUS_FILE, "w") as sf:
            sf.write(f"ranking,{idx+1},{time.time()-t0:.0f}\n")
        
        if (idx+1) % 50 == 0:
            print(f"  Ranking [{idx+1}/{len(all_questions)}]", flush=True)
    
    print(f"Embedding done in {time.time()-t0:.0f}s.", flush=True)

    # ---- QA Generation + Judging ----
    print(f"\nRunning QA gen ({GEN_MODEL}) + judge ({JUDGE_MODEL}) for {len(all_questions)} questions...", flush=True)
    
    results = []
    correct = 0
    total_q = 0
    qtype_correct = {}
    qtype_total = {}
    gen_cost = 0  # tokens

    for idx, (qid, question, qtype, answer_key) in enumerate(all_questions):
        haystack = all_haystack_sessions[qid]
        ranking = question_rankings.get(qid)
        if ranking is None:
            # Skipped or no haystack — add placeholder result
            results.append({
                "question_id": qid, "question": question[:120], "qtype": qtype,
                "answer": str(answer_key), "hypothesis": "[SKIPPED: encoding timeout]",
                "judge_response": "", "correct": False
            })
            total_q += 1
            qtype_total.setdefault(qtype, 0)
            qtype_total[qtype] += 1
            qtype_correct.setdefault(qtype, 0)
            print(f"[{idx+1}/{len(all_questions)}] SKIP | {qtype:25s} | {question[:50]}...", flush=True)
            continue
        
        top_sessions = ranking[:TOP_K]
        
        # Build context with text + timestamp
        sessions_data = []
        for i, si in enumerate(top_sessions):
            text = all_haystack_texts[qid][si]
            # Try to get timestamp from the haystack session object
            sess_obj = haystack[si]
            ts = extract_timestamp(sess_obj) if isinstance(sess_obj, dict) else ""
            sessions_data.append((text, ts))
        
        # Generate
        gen_prompt = build_gen_prompt(question[:500], sessions_data)
        try:
            hypothesis = llm_call(gen_prompt, model=GEN_MODEL, max_tokens=1024, timeout=90)
        except Exception as e:
            hypothesis = f"[GEN ERROR: {e}]"
        
        # Judge
        judge_prompt = get_judge_prompt(qtype, question[:500], str(answer_key), hypothesis)
        try:
            judge_resp = llm_call(judge_prompt, model=JUDGE_MODEL, max_tokens=10, timeout=30)
            is_correct = "yes" in judge_resp.lower()
        except Exception as e:
            judge_resp = f"[JUDGE ERROR: {e}]"
            is_correct = False
        
        if is_correct:
            correct += 1
        total_q += 1
        qtype_correct.setdefault(qtype, 0)
        qtype_total.setdefault(qtype, 0)
        if is_correct:
            qtype_correct[qtype] += 1
        qtype_total[qtype] += 1
        
        results.append({
            "question_id": qid,
            "question": question[:120],
            "qtype": qtype,
            "answer": str(answer_key),
            "hypothesis": hypothesis,
            "judge_response": judge_resp,
            "correct": is_correct
        })
        
        acc = correct / total_q * 100
        elapsed = time.time() - T_START
        print(f"[{idx+1}/{len(all_questions)}] {acc:.1f}% | {'✓' if is_correct else '✗'} | {qtype:25s} | {elapsed:.0f}s | {question[:50]}...", flush=True)
        
        # Write heartbeat every question
        with open(STATUS_FILE, "w") as sf:
            sf.write(f"gen,{idx+1},{correct},{total_q},{elapsed:.0f}\n")
        
        # Save partial every 25
        if (idx+1) % 25 == 0:
            with open(OUTPUT_FILE + ".partial", "w") as f:
                json.dump({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "gen_model": GEN_MODEL,
                    "judge_model": JUDGE_MODEL,
                    "top_k": TOP_K,
                    "total": total_q,
                    "correct": correct,
                    "accuracy": correct / max(total_q, 1),
                    "qtype_accuracy": {k: qtype_correct[k] / qtype_total[k] for k in qtype_correct},
                    "results": results
                }, f, indent=2, default=str)

    # Final
    acc = correct / max(total_q, 1) * 100
    print("\n" + "=" * 70)
    print(f"E2E QA v2 — {GEN_MODEL} gen + {JUDGE_MODEL} judge, top-{TOP_K}")
    print("=" * 70)
    print(f"  Questions: {total_q}/{len(all_questions)}")
    print(f"  Correct:   {correct}/{total_q}")
    print(f"  Accuracy:  {acc:.2f}%")
    print()
    print("  By type:")
    for k in sorted(qtype_correct.keys()):
        ca = qtype_correct[k] / qtype_total[k] * 100
        print(f"    {k:30s}: {ca:5.1f}% ({qtype_correct[k]:3d}/{qtype_total[k]:3d})")
    print()
    print(f"  Time: {time.time()-T_START:.0f}s")
    print("=" * 70)

    with open(OUTPUT_FILE, "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "gen_model": GEN_MODEL,
            "judge_model": JUDGE_MODEL,
            "top_k": TOP_K,
            "retrieval": "chunked_gte_large_500_100_m3",
            "total": total_q,
            "correct": correct,
            "accuracy": round(acc / 100, 4),
            "qtype_accuracy": {k: round(qtype_correct[k] / qtype_total[k], 4) for k in qtype_correct},
            "results": results
        }, f, indent=2, default=str)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
