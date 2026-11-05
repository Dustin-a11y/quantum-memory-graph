#!/usr/bin/env python3
"""Prove (or disprove) supersession on Kong Graph, on a difficulty ladder.

Compares the retrieval path agents actually use — `hybrid_search` — against
`state_aware_search` from `quantum_memory_graph.supersession`, on 50 generated state-tracking cases
laid out in the StateBench difficulty order:

  r-trivial  latest write wins (one attribute, several values)
  aperiodic  same, with similar distractor entities that must not contaminate the answer
  solvable   longer chains, asked in both directions (what is it now / what was it before)
  periodic   one- and two-hop references (the replica's primary is the main server)

Scoring is exact-value containment in the top-1 result. The same deterministic bag-of-words embedder
serves both arms, so no arm gets a better model — the only difference is whether the store knows
which of its own facts has expired.

Usage: python3 qmg-prove-supersession.py [--verbose]
"""
import hashlib
import re
import sys
from datetime import datetime, timedelta

RELEASE = '/home/dt/.qmg-fleet-release/l3-konggraph-alias-20261030T175842'
sys.path.insert(0, RELEASE)
sys.path.insert(0, '/home/dt/qmg-repo')

import numpy as np
from quantum_memory_graph.graph import MemoryGraph
from quantum_memory_graph.search import hybrid_search
from quantum_memory_graph.supersession import StateIndex, state_aware_search, parse_claim

DIMS = 256
BASE_TIME = datetime(2023, 1, 1, 9, 0, 0)


class BagEmbedder:
    """Deterministic hashed bag-of-words. Same weights for every arm, no model download."""

    def encode(self, texts, normalize_embeddings=True):
        out = np.zeros((len(texts), DIMS), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", (text or "").lower()):
                h = int(hashlib.md5(token.encode()).hexdigest()[:8], 16)
                out[i, h % DIMS] += 1.0
            if normalize_embeddings:
                norm = np.linalg.norm(out[i])
                if norm:
                    out[i] /= norm
        return out


def add(graph, text, when):
    mem = graph.add_memory(text, timestamp=when, source="probe")
    mem.embedding = graph.embedder.encode([text])[0]
    return mem


def build_cases():
    """50 cases: (family, question, expected_value, [(text, datetime), ...])."""
    cases = []
    towns = ["Seattle", "Portland", "Austin", "Denver", "Boston", "Chicago", "Miami", "Tucson"]

    # r-trivial: 12 x (three sequential values, ask current)
    for i in range(12):
        entity = "the alpha server" if i < 6 else "the bravo server"
        entity = "%s mk%d" % (entity, i)
        chain = [towns[i % 8], towns[(i + 1) % 8], towns[(i + 2) % 8]]
        store = [("%s is located in %s." % (entity, chain[0]), BASE_TIME),
                 # a stale narrative line with heavy lexical overlap on "located" — the case where
                 # text similarity is actively wrong
                 ("the location note for %s says the located site is %s."
                  % (entity, chain[0]), BASE_TIME + timedelta(days=1)),
                 ("%s is located in %s." % (entity, chain[1]), BASE_TIME + timedelta(days=30)),
                 ("%s is located in %s." % (entity, chain[2]), BASE_TIME + timedelta(days=60))]
        cases.append(("r-trivial", "Where is %s located now?" % entity, chain[2], store))

    # aperiodic: 12 x (target chain + two similar distractors that must not be returned)
    for i in range(12):
        entity = "the gamma unit %d" % i
        distractor_a = "the gamma unit %d spare" % i
        distractor_b = "the gamma unit %d clone" % i
        chain = [towns[(i + 3) % 8], towns[(i + 4) % 8]]
        store = [
            ("%s is located in %s." % (entity, chain[0]), BASE_TIME),
            ("%s is located in %s." % (distractor_a, towns[(i + 5) % 8]), BASE_TIME + timedelta(days=5)),
            ("%s is located in %s." % (entity, chain[1]), BASE_TIME + timedelta(days=30)),
            ("%s is located in %s." % (distractor_b, towns[(i + 6) % 8]), BASE_TIME + timedelta(days=40)),
        ]
        cases.append(("aperiodic", "Where is %s located now?" % entity, chain[1], store))

    # solvable: 13 x (4-step chain; 6 ask current, 7 ask the previous value)
    for i in range(13):
        entity = "the delta node %d" % i
        chain = [towns[(i + j) % 8] for j in range(4)]
        store = [("%s is located in %s." % (entity, v), BASE_TIME + timedelta(days=20 * j))
                 for j, v in enumerate(chain)]
        if i % 2 == 0:
            cases.append(("solvable", "Where is %s located now?" % entity, chain[-1], store))
        else:
            cases.append(("solvable", "Where was %s located before that?" % entity, chain[-2], store))

    # periodic: 13 x (one- or two-hop reference to the entity that actually holds the value)
    for i in range(13):
        main = "the epsilon main %d" % i
        replica = "the epsilon replica %d" % i
        chain = [towns[(i + 2) % 8], towns[(i + 7) % 8]]
        store = [
            ("%s is located in %s." % (main, chain[0]), BASE_TIME),
            ("%s's primary is %s." % (replica, main), BASE_TIME + timedelta(days=10)),
            ("%s is located in %s." % (main, chain[1]), BASE_TIME + timedelta(days=40)),
        ]
        cases.append(("periodic", "Where is %s located now?" % replica, chain[1], store))
    return cases


def build_controls():
    """10 cases where supersession must KEEP ITS HANDS OFF: no state change, or not a state claim."""
    controls = []
    towns = ["Seattle", "Portland", "Austin", "Denver"]
    for i in range(4):
        entity = "the zeta unit %d" % i
        controls.append(("control-single",
                         "Where is %s located now?" % entity, towns[i],
                         [("%s is located in %s." % (entity, towns[i]), BASE_TIME)]))
    for i in range(3):
        entity = "the theta unit %d" % i
        controls.append(("control-restated",
                         "Where is %s located now?" % entity, towns[i % 4],
                         [("%s is located in %s." % (entity, towns[i % 4]), BASE_TIME),
                          ("%s is located in %s." % (entity, towns[i % 4]), BASE_TIME + timedelta(days=9))]))
    for i in range(3):
        entity = "the iota decision %d" % i
        controls.append(("control-nonstate",
                         "What did we decide about %s?" % entity, "ship it on Friday",
                         [("%s: the team decided to ship it on Friday." % entity, BASE_TIME),
                          ("%s was discussed again in the retro." % entity, BASE_TIME + timedelta(days=7))]))
    return controls


def run_case(family, question, expected, store, verbose=False, want_resolution=False):
    graph = MemoryGraph()
    graph._embedder = BagEmbedder()
    for text, when in store:
        add(graph, text, when)

    index = StateIndex.from_graph(graph)
    base = hybrid_search(query=question, graph=graph, k=5)
    aware = state_aware_search(query=question, graph=graph, k=5, index=index)

    def hit(results):
        return bool(results) and expected.lower() in (results[0].get("text") or "").lower()

    base_hit, aware_hit = hit(base), hit(aware)
    resolved = index.resolve(question)
    # history questions resolve to the value that was replaced, current ones to the live value
    target = resolved.get("current") if resolved.get("intent") == "current" else resolved.get("previous")
    resolved_value = ((target or {}).get("value") or "").lower()
    resolution_ok = bool(resolved.get("matched")) and expected.lower() in resolved_value
    if verbose:
        print('  [%s] %s' % (family, question))
        print('      expected: %s' % expected)
        print('      baseline top1: %s' % ((base[0]["text"] if base else "(none)")))
        print('      stateful top1: %s' % ((aware[0]["text"] if aware else "(none)")))
    if want_resolution:
        return base_hit, aware_hit, bool(resolution_ok)
    if verbose and not resolution_ok:
        print('      resolve  : %s' % (resolved_value or '(unmatched)'))
    return base_hit, aware_hit, bool(resolution_ok)


def main():
    verbose = '--verbose' in sys.argv
    cases = build_cases()
    print('supersession probe — %d cases, baseline %s vs state_aware_search\n'
          % (len(cases), 'hybrid_search (the path agent memory injection uses)'))

    # sanity: the parser must recover the claims it is expected to reason over
    parse_failures = 0
    for family, question, expected, store in cases:
        for text, when in store:
            if parse_claim(text, 'x', when) is None:
                parse_failures += 1
    print('claim parsing: %d/%d sentences understood'
          % (sum(len(s) for _, _, _, s in cases) - parse_failures, sum(len(s) for _, _, _, s in cases)))

    by_family = {}
    resolved_ok = 0
    for family, question, expected, store in cases:
        b, a, resolved = run_case(family, question, expected, store, verbose, want_resolution=True)
        d = by_family.setdefault(family, {'n': 0, 'base': 0, 'aware': 0})
        d['n'] += 1
        d['base'] += int(b)
        d['aware'] += int(a)
        resolved_ok += int(resolved)

    order = ['r-trivial', 'aperiodic', 'solvable', 'periodic']
    print('\n%-12s %5s %10s %10s' % ('family', 'n', 'baseline', 'supersession'))
    total = {'n': 0, 'base': 0, 'aware': 0}
    for family in order:
        d = by_family.get(family)
        if not d:
            continue
        print('%-12s %5d %9.3f %10.3f'
              % (family, d['n'], d['base'] / d['n'], d['aware'] / d['n']))
        for k in total:
            total[k] += d[k]
    print('%-12s %5d %9.3f %10.3f'
          % ('ALL', total['n'], total['base'] / total['n'], total['aware'] / total['n']))
    print('\nbaseline %d/%d = %.3f   supersession %d/%d = %.3f   delta %+.3f'
          % (total['base'], total['n'], total['base'] / total['n'],
             total['aware'], total['n'], total['aware'] / total['n'],
             (total['aware'] - total['base']) / total['n']))
    print('state resolution correct on %d/%d ladder cases = %.3f'
          % (resolved_ok, total['n'], resolved_ok / total['n']))

    # Safety: where there is no state change to apply, the resolver must keep its hands off.
    controls = build_controls()
    injected, kept = 0, 0
    for family, question, expected, store in controls:
        graph = MemoryGraph()
        graph._embedder = BagEmbedder()
        for text, when in store:
            add(graph, text, when)
        index = StateIndex.from_graph(graph)
        base = hybrid_search(query=question, graph=graph, k=5)
        aware = state_aware_search(query=question, graph=graph, k=5, index=index)
        if aware and aware[0].get("source") == "supersession-resolution":
            injected += 1
            if verbose:
                print('  [%s] INJECTED on: %s' % (family, question))
        if base and aware and base[0]["id"] == aware[0]["id"]:
            kept += 1
    print('controls (%d): ranking untouched %d/%d, wrong injection %d'
          % (len(controls), kept, len(controls), injected))
    print('PROBE_DONE')


if __name__ == '__main__':
    main()
