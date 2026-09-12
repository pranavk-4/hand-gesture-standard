# Worktree setup (Windows PowerShell). Run from the repo root.
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$parent = Split-Path -Parent $root
Set-Location $root
git worktree add "$parent\wt-hpo" track/hpo-hyperband-bohb
git worktree add "$parent\wt-profiling" track/offline-profiling
git worktree add "$parent\wt-optimization" track/optimization-toolkit
git worktree list
