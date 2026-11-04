#!/usr/bin/env python3
"""Install Kong Graph into a Hermes profile as a plugin, in one command.

What it does, all idempotent and reversible:

1. copies the bundled memory-provider plugin (`qmg_hermes_plugin/`) into
   `<hermes_home>/plugins/memory/qmg/` (or the profile's own `plugins/` dir with --profile-local),
2. sets `memory.provider: qmg` in the profile's config.yaml,
3. registers the packaged MCP server as `mcp.servers.qmg` pointing at the `qmg-mcp` console script,
4. writes the secrets it expects into `<profile>/.env` **only if they are not already there** — it
   never invents or prints a token.

Defaults to a DRY RUN: it reports every change it would make and touches nothing. Pass --apply to
write. A backup of config.yaml is taken before any edit.

    qmg-hermes-install --hermes-home ~/.hermes --profile dk            # preview
    qmg-hermes-install --hermes-home ~/.hermes --profile dk --apply     # do it
"""
import argparse
import hashlib
import os
import shutil
import sys

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML ships with Hermes
    sys.stderr.write('PyYAML is required: pip install pyyaml\n')
    raise SystemExit(2)

PLUGIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hermes_plugin')
PLUGIN_FILES = ('__init__.py', 'config_schema.py', 'plugin.yaml')
REQUIRED_ENV = ('QMG_URL', 'QMG_TOKEN', 'QMG_AGENT')


def load_yaml(path):
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def dump_yaml(path, data):
    tmp = path + '.tmp-konggraph'
    with open(tmp, 'w') as fh:
        yaml.safe_dump(data, fh, sort_keys=False, default_flow_style=False, allow_unicode=True)
    os.replace(tmp, path)


def read_env(path):
    values = {}
    if os.path.exists(path):
        with open(path, errors='replace') as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    values[k.strip()] = v.strip().strip('"').strip("'")
    return values


def plan(args):
    home = os.path.expanduser(args.hermes_home)
    profile_dir = os.path.join(home, 'profiles', args.profile) if args.profile else home
    if args.profile and args.profile == 'default':
        profile_dir = home
    plugin_target = (os.path.join(profile_dir, 'plugins', 'memory', 'qmg')
                     if args.profile_local else
                     os.path.join(home, 'hermes-agent', 'plugins', 'memory', 'qmg'))
    config_path = os.path.join(profile_dir, 'config.yaml')
    env_path = os.path.join(profile_dir, '.env')

    steps, notes = [], []

    missing = [f for f in PLUGIN_FILES if not os.path.exists(os.path.join(PLUGIN_DIR, f))]
    if missing:
        raise SystemExit('packaged plugin incomplete, missing: %s' % ', '.join(missing))
    def _digest(path):
        h = hashlib.sha256()
        with open(path, 'rb') as fh:
            h.update(fh.read())
        return h.hexdigest()

    stale = [f for f in PLUGIN_FILES
             if not os.path.exists(os.path.join(plugin_target, f))
             or _digest(os.path.join(PLUGIN_DIR, f)) != _digest(os.path.join(plugin_target, f))]
    if stale:
        steps.append(('copy plugin files', '%s -> %s (%s)'
                      % (PLUGIN_DIR, plugin_target, ', '.join(stale))))
    else:
        notes.append('plugin files already up to date')

    cfg = load_yaml(config_path)
    mem = cfg.get('memory') or {}
    if mem.get('provider') != 'qmg':
        steps.append(('set memory.provider', 'qmg (was %r)' % mem.get('provider')))
        mem['provider'] = 'qmg'
        cfg['memory'] = mem
    else:
        notes.append('memory.provider already qmg')

    servers = ((cfg.get('mcp') or {}).get('servers') or {})
    entry = servers.get('qmg') or {}
    want_cmd = args.server_command or shutil.which('qmg-mcp') or 'qmg-mcp'
    current = (entry.get('command') if isinstance(entry, dict) else None)
    if current != want_cmd:
        steps.append(('register mcp.servers.qmg', 'command=%s' % want_cmd))
        servers['qmg'] = {'command': want_cmd, 'args': []}
        cfg.setdefault('mcp', {})['servers'] = servers
    else:
        notes.append('mcp.servers.qmg already points at %s' % want_cmd)

    have = read_env(env_path)
    env_missing = [k for k in REQUIRED_ENV if not have.get(k)]
    if env_missing:
        notes.append('env missing in %s: %s (fill them in yourself — this installer never writes '
                     'credentials)' % (env_path, ', '.join(env_missing)))

    return {'profile_dir': profile_dir, 'plugin_target': plugin_target, 'config_path': config_path,
            'env_path': env_path, 'steps': steps, 'notes': notes, 'config': cfg}


def main():
    ap = argparse.ArgumentParser(description='Install Kong Graph as a Hermes memory plugin.')
    ap.add_argument('--hermes-home', default=os.environ.get('HERMES_HOME', '~/.hermes'))
    ap.add_argument('--profile', default='', help='profile name (empty = the default profile)')
    ap.add_argument('--profile-local', action='store_true',
                    help='install the plugin into <profile>/plugins instead of the shared tree')
    ap.add_argument('--server-command', default='', help='override the qmg-mcp command path')
    ap.add_argument('--apply', action='store_true', help='actually write (default: dry run)')
    args = ap.parse_args()

    p = plan(args)
    print('Kong Graph — Hermes plugin install')
    print('  profile dir : %s' % p['profile_dir'])
    print('  plugin      : %s' % p['plugin_target'])
    print('  config      : %s' % p['config_path'])
    print()
    if not p['steps']:
        print('nothing to change — already installed.')
    else:
        for what, detail in p['steps']:
            print('  %s %-26s %s' % ('did  ' if args.apply else 'would', what, detail))
    for note in p['notes']:
        print('  note: %s' % note)

    if not args.apply:
        print('\ndry run — nothing written. re-run with --apply to make these changes.')
        return 0
    if not p['steps']:
        return 0

    os.makedirs(p['plugin_target'], exist_ok=True)
    for f in PLUGIN_FILES:
        shutil.copy2(os.path.join(PLUGIN_DIR, f), os.path.join(p['plugin_target'], f))
    if os.path.exists(p['config_path']):
        backup = p['config_path'] + '.bak-konggraph-install'
        shutil.copy2(p['config_path'], backup)
        print('  backup: %s' % backup)
    dump_yaml(p['config_path'], p['config'])
    print('\napplied. restart the gateway for the memory provider to load:')
    print('  systemctl --user restart hermes-gateway%s'
          % ('@%s' % args.profile if args.profile else ''))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
