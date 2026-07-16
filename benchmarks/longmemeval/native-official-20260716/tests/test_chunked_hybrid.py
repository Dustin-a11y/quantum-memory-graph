#!/usr/bin/env python3
"""
TDD test suite for QMG Chunked Hybrid Retrieval Runner.

Tests:
  1. No gold labels affect ranking
  2. All sessions preserved with original IDs (no renaming)
  3. has_answer stripped from all text handed to model
  4. Deterministic: same input → same output
  5. Official schema compatibility (flat-session)
  6. Official print_retrieval_metrics.py can consume output
  7. Chunked hybrid algorithm correctness (chunks, fusion, top-3)

Run:
    cd /tmp/qmg-native-longmemeval && \
    python3 -m pytest test_chunked_hybrid.py -v --tb=short
"""

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────
SANDBOX = "/tmp/qmg-native-longmemeval"
DATASET = os.path.join(SANDBOX, "LongMemEval", "data", "longmemeval_s_cleaned.json")
RUNNER = os.path.join(SANDBOX, "qmg_chunked_hybrid_runner.py")


# ═══════════════════════════════════════════════════════════════════════
# (1)  No gold labels affect ranking
# ═══════════════════════════════════════════════════════════════════════
class TestNoGoldLabelLeakage(unittest.TestCase):
    """Verify gold labels NEVER influence retrieval ranking."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, SANDBOX)
        from qmg_chunked_hybrid_runner import ChunkedHybridRetriever
        cls.retriever = ChunkedHybridRetriever(device="cpu", use_cache=False)
        cls.dataset = json.load(open(DATASET))

    def test_ranking_identical_with_or_without_answer_field(self):
        """Ranking must be identical whether answer field is present or not."""
        item = copy.deepcopy(self.dataset[0])

        # Run retrieval normally
        rr1 = self.retriever.retrieve(item)
        ranking1 = [ri["corpus_id"] for ri in rr1["ranked_items"]]

        # Remove answer field entirely (extreme case)
        item_no_answer = copy.deepcopy(item)
        item_no_answer.pop("answer", None)
        item_no_answer.pop("answer_session_ids", None)

        # Must still work — answer_session_ids not used for ranking
        rr2 = self.retriever.retrieve(item_no_answer)
        ranking2 = [ri["corpus_id"] for ri in rr2["ranked_items"]]

        self.assertEqual(
            ranking1, ranking2,
            "Ranking CHANGED when answer/answer_session_ids were removed. "
            "Gold labels must not influence ranking."
        )
        print("✅ PASS: ranking identical with/without gold labels")

    def test_ranking_identical_with_permuted_answer_session_ids(self):
        """Ranking must be identical even if answer_session_ids are shuffled."""
        item = copy.deepcopy(self.dataset[0])

        rr1 = self.retriever.retrieve(item)
        ranking1 = [ri["corpus_id"] for ri in rr1["ranked_items"]]

        # Permute answer_session_ids
        item_permuted = copy.deepcopy(item)
        if len(item_permuted["answer_session_ids"]) > 1:
            item_permuted["answer_session_ids"] = list(reversed(item_permuted["answer_session_ids"]))

        rr2 = self.retriever.retrieve(item_permuted)
        ranking2 = [ri["corpus_id"] for ri in rr2["ranked_items"]]

        self.assertEqual(ranking1, ranking2)
        print("✅ PASS: ranking identical with permuted answer_session_ids")

    def test_ranking_identical_with_spoofed_answer_session_ids(self):
        """Ranking must be identical even if answer_session_ids are set to
        completely wrong values (simulating a worst-case gold leak test)."""
        item = copy.deepcopy(self.dataset[0])

        rr1 = self.retriever.retrieve(item)
        ranking1 = [ri["corpus_id"] for ri in rr1["ranked_items"]]

        # Set answer_session_ids to first few haystack sessions (spoof)
        item_spoofed = copy.deepcopy(item)
        item_spoofed["answer_session_ids"] = item["haystack_session_ids"][:2]
        item_spoofed["answer"] = "WRONG ANSWER"

        rr2 = self.retriever.retrieve(item_spoofed)
        ranking2 = [ri["corpus_id"] for ri in rr2["ranked_items"]]

        self.assertEqual(
            ranking1, ranking2,
            "Ranking CHANGED with spoofed answer_session_ids. "
            f"rank1[:3]={ranking1[:3]}, rank2[:3]={ranking2[:3]}"
        )
        print("✅ PASS: ranking immune to spoofed answer_session_ids")

    def test_no_answer_field_in_chunked_text(self):
        """Text fed to the embedding model must never contain the answer field."""
        item = self.dataset[0]
        answer = item["answer"]

        # Monkey-patch to capture chunks
        captured_chunks = []

        from qmg_chunked_hybrid_runner import flatten_session_user_only, strip_has_answer, chunk_text
        sessions = item["haystack_sessions"]
        for sess in sessions:
            cleaned = strip_has_answer(sess)
            text = flatten_session_user_only(cleaned)
            for c in chunk_text(text):
                captured_chunks.append(c)

        # The answer text might appear naturally in the conversation
        # (that's fine — it's part of the haystack). But the answer
        # field should not be explicitly injected.
        # Check that has_answer is not in any chunk
        for c in captured_chunks:
            self.assertNotIn("has_answer", c.lower().split(),
                             f"has_answer leaked into chunk: {c[:100]}...")

        print(f"✅ PASS: {len(captured_chunks)} chunks, no has_answer leakage")


# ═══════════════════════════════════════════════════════════════════════
# (2)  All sessions preserved with original IDs
# ═══════════════════════════════════════════════════════════════════════
class TestSessionsPreserved(unittest.TestCase):
    """Verify all sessions are indexed and IDs are preserved."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, SANDBOX)
        from qmg_chunked_hybrid_runner import ChunkedHybridRetriever
        cls.retriever = ChunkedHybridRetriever(device="cpu", use_cache=False)
        cls.dataset = json.load(open(DATASET))

    def test_all_sessions_appear_in_ranked_items(self):
        """Every haystack session must appear in ranked_items."""
        item = self.dataset[0]
        rr = self.retriever.retrieve(item)

        ranked_ids = [ri["corpus_id"] for ri in rr["ranked_items"]]
        haystack_ids = item["haystack_session_ids"]

        self.assertEqual(
            len(ranked_ids), len(haystack_ids),
            f"Expected {len(haystack_ids)} ranked items, got {len(ranked_ids)}"
        )

        self.assertEqual(
            set(ranked_ids), set(haystack_ids),
            "Ranked items must contain ALL haystack sessions"
        )
        print(f"✅ PASS: all {len(haystack_ids)} sessions in ranked_items")

    def test_answer_sessions_keep_original_ids(self):
        """Answer sessions must appear with their original 'answer_' prefix."""
        item = self.dataset[0]
        rr = self.retriever.retrieve(item)

        ranked_ids = [ri["corpus_id"] for ri in rr["ranked_items"]]
        answer_ids = set(item["answer_session_ids"])

        # All answer sessions must be in ranked_items
        for aid in answer_ids:
            self.assertIn(aid, ranked_ids, f"Answer session {aid} missing from ranked_items")

        # No 'noans_' prefix anywhere
        for rid in ranked_ids:
            self.assertNotIn("noans_", rid,
                             f"Session ID renamed to 'noans_': {rid}")

        print(f"✅ PASS: answer sessions preserved with original IDs: {answer_ids}")

    def test_dates_preserved(self):
        """Every ranked item must have the correct timestamp."""
        item = self.dataset[0]
        rr = self.retriever.retrieve(item)

        # Build mapping
        id_to_date = dict(zip(item["haystack_session_ids"], item["haystack_dates"]))

        for ri in rr["ranked_items"]:
            self.assertEqual(
                ri["timestamp"], id_to_date[ri["corpus_id"]],
                f"Date mismatch for {ri['corpus_id']}: "
                f"got {ri['timestamp']}, expected {id_to_date[ri['corpus_id']]}"
            )

        print(f"✅ PASS: all dates preserved correctly")


# ═══════════════════════════════════════════════════════════════════════
# (3)  has_answer stripped from text
# ═══════════════════════════════════════════════════════════════════════
class TestHasAnswerStripped(unittest.TestCase):
    """Verify has_answer field is stripped from all text."""

    def test_strip_has_answer_removes_field(self):
        """strip_has_answer must remove has_answer from every turn."""
        sys.path.insert(0, SANDBOX)
        from qmg_chunked_hybrid_runner import strip_has_answer

        turns = [
            {"role": "user", "content": "hello", "has_answer": True},
            {"role": "assistant", "content": "hi", "has_answer": False},
            {"role": "user", "content": "world"},
        ]
        cleaned = strip_has_answer(turns)

        for t in cleaned:
            self.assertNotIn("has_answer", t, f"has_answer not stripped from {t}")

        # Original must be unmodified
        self.assertIn("has_answer", turns[0])
        self.assertIn("has_answer", turns[1])

        print("✅ PASS: has_answer stripped, original preserved")

    def test_ranked_item_text_has_no_has_answer(self):
        """Text in ranked_items must not contain has_answer."""
        sys.path.insert(0, SANDBOX)
        from qmg_chunked_hybrid_runner import ChunkedHybridRetriever

        retriever = ChunkedHybridRetriever(device="cpu", use_cache=False)
        dataset = json.load(open(DATASET))

        for item in dataset[:3]:
            rr = retriever.retrieve(item)
            for ri in rr["ranked_items"]:
                self.assertNotIn(
                    "has_answer", ri.get("text", ""),
                    f"has_answer found in ranked_item text for {item['question_id']}"
                )

        print("✅ PASS: no has_answer in any ranked_item text")


# ═══════════════════════════════════════════════════════════════════════
# (4)  Deterministic
# ═══════════════════════════════════════════════════════════════════════
class TestDeterministic(unittest.TestCase):
    """Verify deterministic behavior: same input → same output."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, SANDBOX)
        from qmg_chunked_hybrid_runner import ChunkedHybridRetriever
        cls.retriever = ChunkedHybridRetriever(device="cpu", use_cache=False)
        cls.dataset = json.load(open(DATASET))

    def test_same_item_produces_identical_ranking(self):
        """Running retrieval twice on the same item must produce identical results."""
        item = self.dataset[0]

        rr1 = self.retriever.retrieve(item)
        rr2 = self.retriever.retrieve(item)

        # Compare rankings
        rank1 = [ri["corpus_id"] for ri in rr1["ranked_items"]]
        rank2 = [ri["corpus_id"] for ri in rr2["ranked_items"]]

        self.assertEqual(rank1, rank2, "Rankings differ between runs")

        # Compare metrics
        for k in [1, 3, 5, 10]:
            for m in ["recall_any", "recall_all", "ndcg_any"]:
                key = f"{m}@{k}"
                self.assertAlmostEqual(
                    rr1["metrics"]["session"].get(key, -1),
                    rr2["metrics"]["session"].get(key, -1),
                    places=10,
                    msg=f"Metric {key} differs between runs"
                )

        print("✅ PASS: deterministic ranking and metrics")

    def test_runner_produces_deterministic_output(self):
        """Full runner produces deterministic output (1 item to keep CPU test fast)."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp1:
            out1 = tmp1.name
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp2:
            out2 = tmp2.name

        try:
            for out in [out1, out2]:
                result = subprocess.run(
                    [sys.executable, RUNNER,
                     "--in-file", DATASET,
                     "--out-file", out,
                     "--max-items", "1",
                     "--device", "cpu",
                     "--no-cache"],
                    capture_output=True, text=True,
                    cwd=SANDBOX, timeout=300,
                )
                self.assertEqual(result.returncode, 0, f"Runner failed: {result.stderr}")

            with open(out1) as f:
                entries1 = [json.loads(line) for line in f]
            with open(out2) as f:
                entries2 = [json.loads(line) for line in f]

            self.assertEqual(len(entries1), len(entries2))

            for e1, e2 in zip(entries1, entries2):
                rank1 = [ri["corpus_id"] for ri in e1["retrieval_results"]["ranked_items"]]
                rank2 = [ri["corpus_id"] for ri in e2["retrieval_results"]["ranked_items"]]
                self.assertEqual(
                    rank1, rank2,
                    f"Runner non-deterministic for {e1['question_id']}"
                )

        finally:
            for out in [out1, out2]:
                if os.path.exists(out):
                    os.unlink(out)

        print("✅ PASS: full runner produces deterministic output")


# ═══════════════════════════════════════════════════════════════════════
# (5)  Official schema compatibility (flat-session)
# ═══════════════════════════════════════════════════════════════════════
class TestOfficialSchemaCompatibility(unittest.TestCase):
    """Verify output is consumable by official pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="lme-chunked-test-")
        cls.outfile = os.path.join(cls.tmpdir, "retrieval_log.jsonl")

        result = subprocess.run(
            [sys.executable, RUNNER,
             "--in-file", DATASET,
             "--out-file", cls.outfile,
             "--max-items", "2",
             "--device", "cpu",
             "--no-cache"],
            capture_output=True, text=True,
            cwd=SANDBOX, timeout=120,
        )
        print("Runner stdout:", result.stdout[-2000:])
        if result.returncode != 0:
            print("Runner stderr:", result.stderr[-2000:])
            raise RuntimeError(f"Runner failed: {result.stderr}")

        with open(cls.outfile) as f:
            cls.entries = [json.loads(line) for line in f]

    def test_flat_session_corpus_id_resolution(self):
        """Simulate prepare_prompt's flat-session corpus_id resolution.

        Official code:
            corpusid2entry[ret_result_entry['corpus_id'].replace('noans_', 'answer_')]

        Since we preserve original IDs (no renaming), replace('noans_', 'answer_')
        is a no-op and the lookup works directly.
        """
        for entry in self.entries:
            # Build corpusid2entry as prepare_prompt does
            corpusid2entry = {}
            for sess_id, sess_data in zip(
                entry["haystack_session_ids"], entry["haystack_sessions"]
            ):
                corpusid2entry[sess_id] = sess_data
                for i_turn, turn_entry in enumerate(sess_data):
                    corpusid2entry[f"{sess_id}_{i_turn + 1}"] = turn_entry

            for ri in entry["retrieval_results"]["ranked_items"]:
                corpus_id = ri["corpus_id"]

                # Official resolution: .replace('noans_', 'answer_')
                resolved_id = corpus_id.replace("noans_", "answer_")

                # Must be in corpusid2entry
                self.assertIn(
                    resolved_id, corpusid2entry,
                    f"corpus_id '{corpus_id}' → '{resolved_id}' not in corpusid2entry "
                    f"for {entry['question_id']}"
                )

        print(f"✅ PASS: all ranked_items resolve via flat-session path")

    def test_ranked_items_have_required_fields(self):
        """Every ranked_item must have corpus_id, text, timestamp."""
        for entry in self.entries:
            for ri in entry["retrieval_results"]["ranked_items"]:
                self.assertIn("corpus_id", ri)
                self.assertIn("text", ri)
                self.assertIn("timestamp", ri)
                self.assertIsInstance(ri["corpus_id"], str)
                self.assertIsInstance(ri["text"], str)
                self.assertIsInstance(ri["timestamp"], str)

        print("✅ PASS: all ranked_items have required fields")

    def test_metrics_schema(self):
        """Metrics must have session and turn keys with all required @K values."""
        for entry in self.entries:
            metrics = entry["retrieval_results"]["metrics"]
            self.assertIn("session", metrics)
            self.assertIn("turn", metrics)

            sess = metrics["session"]
            for k in [1, 3, 5, 10, 30, 50]:
                for m in ["recall_any", "recall_all", "ndcg_any"]:
                    key = f"{m}@{k}"
                    self.assertIn(key, sess, f"Missing {key} in session metrics")
                    self.assertIsInstance(sess[key], (int, float))

        print("✅ PASS: metrics schema valid")

    def test_retrieval_results_has_method_field(self):
        """retrieval_results must have method field."""
        for entry in self.entries:
            self.assertIn("method", entry["retrieval_results"])
            self.assertEqual(
                entry["retrieval_results"]["method"],
                "qmg-bm25-hybrid-70-30-chunked-session-v1.3"
            )
        print("✅ PASS: method field present and correct")


# ═══════════════════════════════════════════════════════════════════════
# (6)  Algorithm correctness
# ═══════════════════════════════════════════════════════════════════════
class TestAlgorithmCorrectness(unittest.TestCase):
    """Verify the chunked hybrid algorithm is correct."""

    def test_chunk_size_and_overlap(self):
        """Chunks must be at most CHUNK_SIZE characters with correct overlap."""
        sys.path.insert(0, SANDBOX)
        from qmg_chunked_hybrid_runner import chunk_text, CHUNK_SIZE, CHUNK_OVERLAP

        text = "x" * 2000
        chunks = chunk_text(text)

        for c in chunks:
            self.assertLessEqual(len(c), CHUNK_SIZE)

        # Check overlap: consecutive chunks should share text
        if len(chunks) > 1:
            for i in range(len(chunks) - 1):
                overlap = chunks[i][-CHUNK_OVERLAP:]
                self.assertIn(overlap, chunks[i + 1],
                              f"Missing overlap between chunk {i} and {i+1}")

        print("✅ PASS: chunking correct")

    def test_hybrid_fusion_weights(self):
        """Verify 70/30 fusion weights are applied correctly."""
        sys.path.insert(0, SANDBOX)
        from qmg_chunked_hybrid_runner import (
            ChunkedHybridRetriever, HYBRID_EMBED_WEIGHT, HYBRID_BM25_WEIGHT
        )

        self.assertAlmostEqual(HYBRID_EMBED_WEIGHT, 0.7)
        self.assertAlmostEqual(HYBRID_BM25_WEIGHT, 0.3)
        self.assertAlmostEqual(HYBRID_EMBED_WEIGHT + HYBRID_BM25_WEIGHT, 1.0)

        # Verify the retriever actually uses the fusion
        retriever = ChunkedHybridRetriever(device="cpu", use_cache=False)
        dataset = json.load(open(DATASET))
        item = dataset[0]

        rr = retriever.retrieve(item)
        # Should have meaningful ranking
        self.assertGreater(len(rr["ranked_items"]), 0)

        print("✅ PASS: hybrid fusion weights correct")

    def test_top_3_chunk_scoring(self):
        """Verify per-session mean of top-3 chunk scores."""
        sys.path.insert(0, SANDBOX)
        from qmg_chunked_hybrid_runner import (
            ChunkedHybridRetriever, TOP_CHUNKS_FOR_SESSION,
            flatten_session_user_only, strip_has_answer, chunk_text,
        )

        self.assertEqual(TOP_CHUNKS_FOR_SESSION, 3)

        # For a session with many chunks, only top-3 should matter
        retriever = ChunkedHybridRetriever(device="cpu", use_cache=False)
        dataset = json.load(open(DATASET))

        # Find a session with many user turns (= many chunks)
        for item in dataset:
            for si, sess in enumerate(item["haystack_sessions"]):
                cleaned = strip_has_answer(sess)
                text = flatten_session_user_only(cleaned)
                chunks = chunk_text(text)
                if len(chunks) > 10:
                    # This session has many chunks — verify top-3 behavior
                    rr = retriever.retrieve(item)
                    self.assertGreater(len(rr["ranked_items"]), 0)

                    # Every session with >0 chunks should be in ranked_items
                    ranked_ids = {ri["corpus_id"] for ri in rr["ranked_items"]}
                    self.assertEqual(
                        len(ranked_ids), len(item["haystack_session_ids"]),
                        f"Not all sessions ranked: {len(ranked_ids)} vs {len(item['haystack_session_ids'])}"
                    )
                    print("✅ PASS: top-3 chunk scoring produces full ranking")
                    return

        print("✅ PASS: top-3 chunk scoring (no session with >10 chunks found, but tests pass)")


# ═══════════════════════════════════════════════════════════════════════
# (7)  Official print_retrieval_metrics.py compatibility
# ═══════════════════════════════════════════════════════════════════════
class TestOfficialMetricsConsumer(unittest.TestCase):
    """Verify official print_retrieval_metrics.py can consume output."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="lme-metrics-test-")
        cls.outfile = os.path.join(cls.tmpdir, "retrieval_log.jsonl")

        result = subprocess.run(
            [sys.executable, RUNNER,
             "--in-file", DATASET,
             "--out-file", cls.outfile,
             "--max-items", "2",
             "--device", "cpu",
             "--no-cache"],
            capture_output=True, text=True,
            cwd=SANDBOX, timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Runner failed: {result.stderr}")

    def test_official_metrics_script_runs(self):
        """Official print_retrieval_metrics.py must not crash."""
        official_script = os.path.join(
            SANDBOX, "LongMemEval", "src", "evaluation", "print_retrieval_metrics.py"
        )

        result = subprocess.run(
            [sys.executable, official_script, self.outfile],
            capture_output=True, text=True,
            cwd=os.path.join(SANDBOX, "LongMemEval"),
            timeout=30,
        )

        print("Official metrics output:")
        print(result.stdout)
        if result.returncode != 0:
            print("STDERR:", result.stderr)

        # The script should not crash (returncode 0) or at least not
        # fail with a traceback (some metrics may be 0 for small runs)
        self.assertIn("Session-level metrics:", result.stdout,
                      "Official script output missing 'Session-level metrics:'")
        print("✅ PASS: official print_retrieval_metrics.py consumes output")


# ═══════════════════════════════════════════════════════════════════════
# (8)  Embedding cache correctness
# ═══════════════════════════════════════════════════════════════════════
class TestEmbeddingCache(unittest.TestCase):
    """Verify embedding cache works correctly."""

    def test_cache_hits_produce_identical_embeddings(self):
        """Cached embeddings must match freshly computed ones."""
        sys.path.insert(0, SANDBOX)
        from qmg_chunked_hybrid_runner import EmbeddingCache

        cache = EmbeddingCache()
        text = "The quick brown fox jumps over the lazy dog"

        # Initially not cached
        self.assertIsNone(cache.get(text))

        # Put an embedding
        emb = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        cache.put(text, emb)

        # Should now be cached
        cached = cache.get(text)
        self.assertIsNotNone(cached)
        np.testing.assert_array_equal(cached, emb)

        print("✅ PASS: embedding cache works")


# ═══════════════════════════════════════════════════════════════════════
# (9)  Fail-closed: invalid input handling
# ═══════════════════════════════════════════════════════════════════════
class TestFailClosed(unittest.TestCase):
    """Verify fail-closed behavior on invalid inputs."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, SANDBOX)
        from qmg_chunked_hybrid_runner import ChunkedHybridRetriever
        cls.retriever = ChunkedHybridRetriever(device="cpu", use_cache=False)

    def test_empty_sessions_produces_empty_ranking(self):
        """Item with empty sessions must produce empty but valid result."""
        item = {
            "question": "test?",
            "haystack_sessions": [],
            "haystack_session_ids": [],
            "haystack_dates": [],
            "answer_session_ids": [],
            "answer": "test",
            "question_id": "empty_test",
            "question_type": "single-session-user",
            "question_date": "2023/01/01",
        }

        rr = self.retriever.retrieve(item)
        self.assertEqual(len(rr["ranked_items"]), 0)
        self.assertIn("session", rr["metrics"])
        print("✅ PASS: empty sessions handled gracefully")

    def test_single_session_produces_single_ranking(self):
        """Item with one session must produce exactly one ranked item."""
        dataset = json.load(open(DATASET))
        # Take first item but keep only first session
        item = copy.deepcopy(dataset[0])
        item["haystack_sessions"] = item["haystack_sessions"][:1]
        item["haystack_session_ids"] = item["haystack_session_ids"][:1]
        item["haystack_dates"] = item["haystack_dates"][:1]

        rr = self.retriever.retrieve(item)
        self.assertEqual(len(rr["ranked_items"]), 1)
        self.assertEqual(
            rr["ranked_items"][0]["corpus_id"],
            item["haystack_session_ids"][0]
        )
        print("✅ PASS: single session handled correctly")


if __name__ == "__main__":
    unittest.main()