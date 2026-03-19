# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is the [Mila cluster technical documentation](https://docs.mila.quebec/), built with **MkDocs + Material theme**. It was recently migrated from Sphinx.

## Common Commands

```bash
# Install dependencies
uv sync
uv run pre-commit install

# Serve locally with live reload (http://127.0.0.1:8000/)
uv run mkdocs serve --livereload

# Build static site
uv run mkdocs build

# Run pre-commit hooks (linting + preprocessing)
uv run pre-commit run --all-files

# Run integration tests (requires Mila cluster / SLURM access)
uv run --all-groups pytest -v -n 4
```

## Architecture

### Documentation Framework

- **MkDocs** with the **Material theme** (`mkdocs.yml`)
- Navigation is defined in `docs/README.md` via the `literate-nav` plugin — this file is the source of truth for site structure, not `mkdocs.yml`
- Custom HTML/CSS overrides live in `overrides/` (extends Material base templates)
- Shared abbreviations in `includes/abbreviations.md` are automatically injected via `pymdownx.snippets`

### Content Organization

All documentation markdown lives in `docs/`. The main sections are:

- **How-tos and Guides** — user-facing guides (quick start, jobs, MFA, containers, etc.)
- **Systems and Services** — infrastructure, storage, data policies
- **Minimal Examples** (`docs/examples/`) — runnable code with SLURM integration tests
- **General Theory** — cluster basics, Unix, batch scheduling

### Pre-processing Pipeline

Before build, `docs/examples/preprocess.py` must run to:
- Generate `.diff` files from `before`/`after` code example pairs
- Inline content and rewrite links for GitHub-compatible README files

This runs automatically via the pre-commit hook and in ReadTheDocs' pre-build step (`.readthedocs.yaml`). If you add or modify code examples under `docs/examples/`, run it manually:

```bash
uv run --script docs/examples/preprocess.py
```

### CI/CD

- `build_test.yml` — builds the docs on PRs
- `lint.yml` — runs pre-commit hooks
- `tests.yml` — runs SLURM integration tests on the actual Mila cluster (requires cluster runner)
- `prod.yml` — deploys to production on merge to master
