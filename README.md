# Retrieval Under Stress

How do BM25, a fixed dense retriever and their hybrid differ in retrieval quality and degradation under controlled query noise?

## Current Scope

- Python package
- configuration validation
- CLI
- synthetic test data 
- automated checks

Retrieval and benchmark evaluation are not implemented yet

## Setup

Run commands from the repository root.

```bash 
uv sync --locked
````


# Configuration check

```bash
uv run retrieval-lab doctor
````

## Verification

```bash
uv runn ruff check .
uv run ruff format --check .
uv run pytest
```