# Public Repo Policy

This repository is public. The **code** is public; the **data** is not.

## Never commit agent graph data

- `*.pkl`, `*.pkl.bak` — serialized graph state
- `graph.json` — full graph export (any path)
- `.qmg-data/`, `bootstrap_receipt.json` — local runtime data

A memory pair or a provider token must never reach a commit. Local agent data
lives outside the repo in `/home/dt/.qmg-*` and stays there.

## Two guards, two different jobs

| Guard | Covers | Where |
|---|---|---|
| `.github/workflows/secret-scan.yml` (gitleaks) | secrets: keys, tokens, private keys | CI |
| `.github/workflows/data-artifact-guard.yml` | graph data: `.pkl`, `graph.json` | CI |

gitleaks does not catch graph data — a pickle is not shaped like a secret.
The data-artifact guard runs `git ls-files` and fails if a `.pkl` or a file
named `graph.json` is tracked. `.gitignore` stops `git add` of new files; the
guard also catches files already in the index.

## Optional pre-commit hook

```sh
ln -sf ../../scripts/check_data_artifacts.sh .git/hooks/pre-commit
```

Manual run: `sh scripts/check_data_artifacts.sh`. On failure, untrack while
keeping your local copy: `git rm --cached path/to/file.pkl`.
