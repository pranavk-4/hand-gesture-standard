#!/usr/bin/env bash
# Worktree setup (Linux/macOS). Run from the repo root.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PARENT="$(dirname "$ROOT")"
cd "$ROOT"
git worktree add "$PARENT/wt-hpo" track/hpo-hyperband-bohb
git worktree add "$PARENT/wt-profiling" track/offline-profiling
git worktree add "$PARENT/wt-optimization" track/optimization-toolkit
git worktree list
