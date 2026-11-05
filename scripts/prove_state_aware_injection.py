#!/usr/bin/env python3
"""Prove state-aware injection end to end: real plugin -> real server -> real graph.

Stores two facts that supersede one another in dk's live graph, then asks the ACTUAL Hermes memory
plugin (plugins/memory/qmg, the thing that injects context into every turn) what it would inject for
"where is the probe rack located now". The current fact must come first, and if the stale fact still
appears it must be labelled as superseded.

Cleans up after itself: every probe memory is deleted by id through /delete-by-ids.
"""
import json
import os
import subprocess
import sys
import urllib.request

CONFIG = '/home/dt/.hermes/profiles/dk/qmg-memory/config.json'
HERMES_PY = '/home/dt/.hermes/hermes-agent/venv/bin/python'
SUBJECT = 'the probe rack'
FACTS = ['%s is located in Seattle.' % SUBJECT, '%s is located in Austin.' % SUBJECT]

_PROVIDER_CONFIG = json.load(open(CONFIG))
URL = _PROVIDER_CONFIG['qmg_url']


def token():
    """The provider's own credential, read from its profile config — never printed."""
    value = _PROVIDER_CONFIG.get('auth_token') or ''
    if not value:
        raise SystemExit('no auth_token in %s' % CONFIG)
    return value


def call(path, payload=None, method=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(URL + path, data=data, method=method or ('POST' if data else 'GET'),
                                 headers={'Content-Type': 'application/json',
                                          'Authorization': 'Bearer %s' % token()})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def main():
    checks = []

    def check(name, ok, detail=''):
        checks.append(ok)
        print('  %-58s %s %s' % (name, 'PASS' if ok else 'FAIL', detail))

    before = call('/')['graph']['nodes']
    ids = []
    for fact in FACTS:
        res = call('/store', {'text': fact, 'source': 'supersession-injection-proof'})
        ids.append(res.get('id') or res.get('memory_id'))
    after = call('/')['graph']['nodes']
    check('probe facts stored', all(ids) and after >= before + 2, 'nodes %d -> %d' % (before, after))
    print('  ids: %s' % ids)

    # 1. the route itself
    aware = call('/search', {'query': 'where is %s located now' % SUBJECT, 'k': 6,
                             'state_aware': True})
    top = aware['results'][0]['text'] if aware['results'] else ''
    check('/search state_aware puts the live value first', 'Austin' in top, top[:60])

    # 2. the plugin that actually injects memory, driven in-process
    # Drive the provider exactly as Hermes does: it reads its own profile config, including the
    # token, so no secret passes through this script.
    script = r'''
import json, sys
sys.path.insert(0, "/home/dt/.hermes/hermes-agent")
from plugins.memory.qmg import QMGMemoryProvider

cfg = json.load(open("/home/dt/.hermes/profiles/dk/qmg-memory/config.json"))
prov = QMGMemoryProvider()
prov._config = cfg
prov._qmg_url = cfg["qmg_url"]
prov._agent_name = cfg.get("agent_name", "dk")
prov._enabled = True
print("PROVIDER_OK", type(prov).__name__, prov._qmg_url)
print("PREFETCH_START")
print(prov.prefetch("where is the probe rack located now"))
print("PREFETCH_END")
'''
    r = subprocess.run([HERMES_PY, '-c', script], capture_output=True, text=True, timeout=180)
    out = r.stdout
    if 'PROVIDER_OK' not in out:
        print(r.stdout[-800:])
        print(r.stderr[-1500:])
    check('real plugin constructs', 'PROVIDER_OK' in out)
    block = out.split('PREFETCH_START')[-1].split('PREFETCH_END')[0] if 'PREFETCH_START' in out else ''
    print('\n--- what the agent would actually see ---')
    print(block.strip())
    print('--- end ---\n')
    first_line = next((l for l in block.splitlines() if l.startswith('- ')), '')
    check('injected context leads with the current value', 'Austin' in first_line, first_line[:80])
    stale = [l for l in block.splitlines() if 'Seattle' in l]
    if stale:
        check('stale fact is labelled, not silently dropped',
              any('superseded' in l.lower() for l in stale), stale[0][:80])
    else:
        check('stale fact is labelled, not silently dropped', True, '(not injected at k=6)')

    # 3. clean up
    deleted = call('/delete-by-ids', {'ids': [i for i in ids if i], 'dry_run': False})
    final = call('/')['graph']['nodes']
    check('probe facts removed', final <= after, 'nodes %d -> %d, delete=%s'
          % (after, final, json.dumps(deleted)[:80]))

    print('\n%d/%d checks pass' % (sum(1 for c in checks if c), len(checks)))
    print('INJECTION_PROOF_DONE' if all(checks) else 'INJECTION_PROOF_FAILED')
    return 0 if all(checks) else 1


if __name__ == '__main__':
    raise SystemExit(main())
