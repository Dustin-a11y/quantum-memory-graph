#!/usr/bin/env python3
"""
LongMemEval End-to-End QA — chunked retrieval + DeepSeek generation + judge.

Pre-computes all session embeddings once, then reuses for all 500 questions.
Runs on Spark with CUDA for fast embedding.

DK 🦍 — May 28, 2026
"""
import json, os, sys, time, math
from datetime import datetime, timezone
import numpy as np
import requests

DATA_PATH = "/home/dt/projects-shared/LongMemEval/data/longmemeval_s_cleaned.json"
OUTPUT_DIR = "/home/dt/qmg-v1/benchmarks"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "longmemeval_e2e_qa_results.json")
TOP_K = 5
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

DEEPSEEK_KEY = "REDACTED_DEEPSEEK_KEY"
DEEPSEEK_URL = "https://api.deepseek.com/v1"
GEN_MODEL = "deepseek-v4-pro"
JUDGE_MODEL = "deepseek-v4-pro"

# ---- Helpers ----

def llm_call(messages, model=GEN_MODEL, max_tokens=512, temperature=0):
    resp = requests.post(
        f"{DEEPSEEK_URL}/chat/completions",
        headers={"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature},
        timeout=30
    )
    return resp.json()["choices"][0]["message"]["content"].strip()

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

def build_gen_prompt(question, sessions_text):
    return [
        {"role": "system", "content": "You are a helpful assistant with access to the user's chat history. Answer the question based ONLY on the provided conversation sessions. If the information is not in the sessions, say you cannot answer. Be precise and concise."},
        {"role": "user", "content": f"Relevant conversation sessions:\n\n{sessions_text}\n\nQuestion: {question}\n\nAnswer the question based on the sessions above."}
    ]

def get_judge_prompt(qtype, question, answer, response):
    abstention = "_abs" in str(question)
    if abstention:
        return [{"role": "user", "content": f"I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {question}\n\nExplanation: {answer}\n\nModel Response: {response}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."}]

    templates = {
        'single-session-user': "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
        'single-session-assistant': "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
        'multi-session': "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
        'temporal-reasoning': "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
        'knowledge-update': "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
        'single-session-preference': "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
    }
    t = templates.get(qtype, templates['single-session-user'])
    return [{"role": "user", "content": t.format(question, answer, response)}]

# ---- Main ----

def main():
    T_START = time.time()
    print("Loading LongMemEval data...", flush=True)
    with open(DATA_PATH) as f: ref_data = json.load(f)

    # Build index by question_id
    ref_by_qid = {}
    for item in ref_data:
        ref_by_qid[item["question_id"]] = item

    print(f"Loaded {len(ref_data)} questions. Loading gte-large...", flush=True)
    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("thenlper/gte-large", device=device)
    print(f"Model on {device}. Pre-computing all session embeddings...", flush=True)

    # Pre-compute chunk embeddings for ALL sessions across ALL questions
    # Build unique session set per question, but batch all at once
    all_questions = []
    all_haystack_texts = {}  # question_id -> [session_texts]
    all_haystack_sessions = {}
    
    for item in ref_data:
        qid = item["question_id"]
        haystack = item.get("haystack_sessions", [])
        all_haystack_sessions[qid] = haystack
        texts = [flatten_session(s) for s in haystack]
        all_haystack_texts[qid] = texts
        all_questions.append((qid, item["question"], item.get("question_type", "unknown"), item.get("answer", "")))
    
    # Cache: for each question, pre-compute chunk-level scores
    # For efficiency, batch-embed all chunks per question
    print("Computing retrieval scores (chunked gte-large)...", flush=True)
    question_rankings = {}
    t0 = time.time()
    
    for idx, (qid, question, qtype, answer_key) in enumerate(all_questions):
        haystack = all_haystack_sessions[qid]
        if not haystack or len(haystack) < 3:
            question_rankings[qid] = (None, None, None)
            continue
        
        # Chunk
        all_chunks = []
        chunk_to_session = []
        for si, sess in enumerate(haystack):
            for c in chunk_text(all_haystack_texts[qid][si]):
                all_chunks.append(c)
                chunk_to_session.append(si)
        
        # Encode
        q_emb = model.encode([question], normalize_embeddings=True, show_progress_bar=False)[0]
        chunk_embs = model.encode(all_chunks, normalize_embeddings=True, batch_size=256, show_progress_bar=False)
        chunk_scores = chunk_embs @ q_emb
        
        session_chunk_scores = {}
        for ci, si in enumerate(chunk_to_session):
            if si not in session_chunk_scores:
                session_chunk_scores[si] = []
            session_chunk_scores[si].append(float(chunk_scores[ci]))
        
        session_scores = {}
        for si, scores in session_chunk_scores.items():
            sorted_s = sorted(scores, reverse=True)
            top_n = min(3, len(sorted_s))
            session_scores[si] = float(np.mean(sorted_s[:top_n]))
        
        ranking = sorted(range(len(haystack)), key=lambda i: session_scores.get(i, -1.0), reverse=True)
        question_rankings[qid] = (ranking, session_scores, len(all_chunks))
        
        if (idx+1) % 50 == 0:
            print(f"  Embedding [{idx+1}/{len(all_questions)}]", flush=True)
    
    print(f"Embedding done in {time.time()-t0:.0f}s. Running QA gen+judge...", flush=True)

    # ---- QA Generation + Judging ----
    results = []
    correct = 0
    total_q = 0
    qtype_correct = {}
    qtype_total = {}

    for idx, (qid, question, qtype, answer_key) in enumerate(all_questions):
        haystack = all_haystack_sessions[qid]
        ranking, _, _ = question_rankings[qid]
        if ranking is None:
            continue
        
        top_sessions = ranking[:TOP_K]
        
        # Build context
        context_parts = []
        for i, si in enumerate(top_sessions):
            context_parts.append(f"--- Session {i+1} ---\n{all_haystack_texts[qid][si]}")
        sessions_text = "\n\n".join(context_parts)
        
        # Generate
        gen_prompt = build_gen_prompt(question[:500], sessions_text[:8000])
        try:
            hypothesis = llm_call(gen_prompt)
        except Exception as e:
            hypothesis = f"[GEN ERROR: {e}]"
        
        # Judge
        judge_prompt = get_judge_prompt(qtype, question[:500], answer_key, hypothesis)
        try:
            judge_resp = llm_call(judge_prompt, model=JUDGE_MODEL, max_tokens=10)
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
            "answer": answer_key,
            "hypothesis": hypothesis,
            "judge_response": judge_resp,
            "correct": is_correct
        })
        
        acc = correct / total_q * 100
        elapsed = time.time() - T_START
        print(f"[{idx+1}/{len(all_questions)}] {acc:.1f}% | {'✓' if is_correct else '✗'} | {qtype} | {elapsed:.0f}s | Q: {question[:60]}...", flush=True)
        
        # Save every 25
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
    print(f"END-TO-END QA — {GEN_MODEL} gen + {JUDGE_MODEL} judge, top-{TOP_K} chunked retrieval")
    print("=" * 70)
    print(f"  Questions: {total_q}/{len(all_questions)}")
    print(f"  Correct:   {correct}/{total_q}")
    print(f"  Accuracy:  {acc:.2f}%")
    print()
    print("  By type:")
    for k in sorted(qtype_correct.keys()):
        ca = qtype_correct[k] / qtype_total[k] * 100
        print(f"    {k}: {ca:.1f}% ({qtype_correct[k]}/{qtype_total[k]})")
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
