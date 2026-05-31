#!/usr/bin/env python3
"""
LongMemEval Judge Re-Run — re-judge existing hypotheses with correct model.
Reads existing e2e QA results (hypotheses already generated), re-judges all 500.

DK 🦍 — May 29, 2026
"""
import json, os, sys, time
from datetime import datetime, timezone

RESULTS_FILE = "/home/dt/qmg-v1/benchmarks/longmemeval_e2e_qa_results.json"
OUTPUT_FILE = "/home/dt/qmg-v1/benchmarks/longmemeval_e2e_qa_results_judged.json"

DEEPSEEK_KEY = os.environ.get('DEEPSEEK_API_KEY', '')
DEEPSEEK_URL = "https://api.deepseek.com/v1"
JUDGE_MODEL = "deepseek-chat"  # FIXED: was deepseek-v4-pro which returns empty

import requests

def llm_call(messages, model=JUDGE_MODEL, max_tokens=10, temperature=0, timeout=30):
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
            content = j["choices"][0]["message"]["content"].strip()
            if content:
                return content
            # Empty content with 200 — retry
            print(f"  [WARN] Empty content attempt {attempt+1}", flush=True)
        except Exception as e:
            last_e = e
            print(f"  [WARN] Attempt {attempt+1} failed: {e}", flush=True)
            time.sleep(2 ** attempt)
    return f"[JUDGE FAILED: {last_e}]" if last_e else "[JUDGE FAILED: empty content]"

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

def main():
    print(f"Loading existing results from {RESULTS_FILE}...", flush=True)
    with open(RESULTS_FILE) as f:
        data = json.load(f)

    results = data["results"]
    print(f"Loaded {len(results)} results. Re-judging all...", flush=True)

    T_START = time.time()
    correct = 0
    total_q = 0
    qtype_correct = {}
    qtype_total = {}

    for idx, r in enumerate(results):
        qid = r["question_id"]
        question = r["question"]
        qtype = r["qtype"]
        answer = r["answer"]
        hypothesis = r["hypothesis"]

        # Build judge prompt
        judge_prompt = get_judge_prompt(qtype, question, answer, hypothesis)
        
        # Call judge
        judge_resp = llm_call(judge_prompt)
        is_correct = "yes" in judge_resp.lower()

        # Update
        r["judge_response"] = judge_resp
        r["correct"] = is_correct

        if is_correct:
            correct += 1
        total_q += 1
        qtype_correct.setdefault(qtype, 0)
        qtype_total.setdefault(qtype, 0)
        if is_correct:
            qtype_correct[qtype] += 1
        qtype_total[qtype] += 1

        acc = correct / total_q * 100
        elapsed = time.time() - T_START
        print(f"[{idx+1}/{len(results)}] {acc:.1f}% | {'✓' if is_correct else '✗'} | {qtype} | {elapsed:.0f}s | Q: {question[:60]}...", flush=True)

        # Save partial every 25
        if (idx+1) % 25 == 0:
            with open(OUTPUT_FILE + ".partial", "w") as f:
                json.dump({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "gen_model": data.get("gen_model", "deepseek-v4-pro"),
                    "judge_model": JUDGE_MODEL,
                    "top_k": data.get("top_k", 5),
                    "total": total_q,
                    "correct": correct,
                    "accuracy": correct / max(total_q, 1),
                    "qtype_accuracy": {k: qtype_correct[k] / qtype_total[k] for k in qtype_correct},
                    "results": results
                }, f, indent=2, default=str)

    # Final
    acc = correct / max(total_q, 1) * 100
    print("\n" + "=" * 70)
    print(f"RE-JUDGED — {JUDGE_MODEL} judge, {total_q} questions")
    print("=" * 70)
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

    # Update original record
    data["correct"] = correct
    data["accuracy"] = round(acc / 100, 4)
    data["judge_model"] = JUDGE_MODEL
    data["qtype_accuracy"] = {k: round(qtype_correct[k] / qtype_total[k], 4) for k in qtype_correct}

    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\nSaved to {OUTPUT_FILE}", flush=True)

if __name__ == "__main__":
    main()
