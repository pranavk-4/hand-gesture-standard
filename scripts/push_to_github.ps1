# Push this repo to GitHub and protect `main`. Run once (maintainer).
# 1. Create an empty GitHub repo (no README/license), e.g. hand-gesture-standard.
# 2. Run this script with its URL:
#      .\scripts\push_to_github.ps1 https://github.com/<org>/hand-gesture-standard.git

param([Parameter(Mandatory=$true)][string]$RemoteUrl)
Set-Location (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
git remote add origin $RemoteUrl
git push -u origin main track/hpo-hyperband-bohb track/offline-profiling track/optimization-toolkit --tags
Write-Host "Done. Now on GitHub: Settings > Branches > Add rule for 'main' (require PR + status check 'smoke')."
