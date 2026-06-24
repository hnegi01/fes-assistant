#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Refreshing PySisense tool registry ==="
echo "Repo root: ${REPO_ROOT}"
echo

cd "${REPO_ROOT}"

echo "[0/2] Installing pinned pysisense (from requirements.txt)..."
pip install -r requirements.txt

echo "[1/2] Building tool registry from SDK..."
python -m scripts.01_build_registry_from_sdk

echo "[2/2] Adding LLM examples and building hierarchical registry..."
python -m scripts.02_add_llm_examples_to_registry

echo
echo "Generated files:"
echo "  - config/tools.registry.json"
echo "  - config/tools.registry.with_examples.json"
echo "  - config/registry/  (hierarchical, with examples)"
echo
echo "=== Done ==="
