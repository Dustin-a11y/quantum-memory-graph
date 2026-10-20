#!/bin/sh
# check_data_artifacts.sh — refuse to commit QMG graph data artifacts.
#
# QMG graph data (pickles and graph.json exports) contains agent memory and
# must never enter this public repository. This script inspects the git index
# via `git ls-files` and exits 1 if any forbidden path is tracked.
# It stages nothing and changes nothing.
#
# Install as a local pre-commit hook (optional, per clone):
#   ln -sf ../../scripts/check_data_artifacts.sh .git/hooks/pre-commit
#
# Run manually:
#   sh scripts/check_data_artifacts.sh
#
# Exit codes: 0 = clean, 1 = forbidden artifact tracked, 2 = not a git repo.

set -eu

if ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo "check_data_artifacts: not a git repository" >&2
    exit 2
fi

bad=$(git ls-files | grep -E '(^|/)graph\.json$|\.pkl$' || true)

if [ -n "$bad" ]; then
    echo "check_data_artifacts: refusing to commit QMG graph data artifacts." >&2
    echo "These files hold agent graph data and must not be tracked in a public repo:" >&2
    echo "$bad" | sed 's/^/  /' >&2
    echo "" >&2
    echo "Fix: git rm --cached <path>   (keeps your local copy)" >&2
    echo "See docs/PUBLIC-REPO-POLICY.md" >&2
    exit 1
fi

exit 0
