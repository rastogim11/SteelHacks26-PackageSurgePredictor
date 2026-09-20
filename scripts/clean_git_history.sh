#!/usr/bin/env bash
#
# Purge the API key and the student PII from git history, then force push.
#
#   bash scripts/clean_git_history.sh
#
# READ THIS BEFORE RUNNING.
#
# This rewrites every commit. Each commit gets a new SHA, so the remote history
# is replaced rather than added to, and anyone else holding a clone must delete
# it and clone again - their old clone can reintroduce what this removes.
#
# This does NOT un-expose anything already fetched. The repository has been
# public with a live key in it, and public repos are continuously scraped.
# ROTATE THE ELEVENLABS KEY FIRST. This script is the cleanup, not the fix.
#
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

PATHS_TO_PURGE=(
  "elevenLabs.py"                    # held a live ElevenLabs API key
  "packagestats.csv"                 # recipient names
  "packagestats.xlsx"                # recipient names
  "files/packagestats_cleaned.xlsx"  # recipient names
  "searchresults(1).csv"             # recipient names
)

echo "Repository: $REPO_ROOT"
echo
echo "The following paths will be removed from ALL commits:"
printf '  %s\n' "${PATHS_TO_PURGE[@]}"
echo
echo "Every commit SHA changes. The remote is overwritten with --force."
echo "Anyone else with a clone must re-clone afterwards."
echo
read -r -p "Type REWRITE to continue: " confirm
[ "$confirm" = "REWRITE" ] || { echo "Aborted."; exit 1; }

# git-filter-repo is not part of git. Prefer it over filter-branch, which is
# slow and leaves reflog and stash copies behind.
if ! command -v git-filter-repo >/dev/null 2>&1; then
  echo
  echo "git-filter-repo is not installed. Install one of:"
  echo "    brew install git-filter-repo"
  echo "    python3 -m pip install --user git-filter-repo"
  exit 1
fi

# A mirror clone beside the repo, so a bad rewrite is recoverable.
BACKUP="../SteelHacks26-backup-$(date +%Y%m%d-%H%M%S).git"
echo
echo "Backing up to $BACKUP"
git clone --mirror . "$BACKUP"

REMOTE_URL="$(git remote get-url origin)"

echo "Rewriting history..."
ARGS=()
for p in "${PATHS_TO_PURGE[@]}"; do
  ARGS+=(--path "$p")
done
git filter-repo --invert-paths "${ARGS[@]}" --force

# filter-repo drops the remote on purpose, so a force push cannot happen by
# reflex. Putting it back is the deliberate step.
git remote add origin "$REMOTE_URL" 2>/dev/null || git remote set-url origin "$REMOTE_URL"

echo
echo "History rewritten. Local checks:"
echo -n "  paths still tracked: "
git ls-files | grep -Ei 'packagestats|searchresults|elevenLabs' || echo "none"
echo -n "  blobs still in history: "
git rev-list --objects --all | grep -Ei 'packagestats|searchresults|elevenLabs' || echo "none"

cat <<'EOF'

Nothing has been pushed yet. To publish the rewrite:

    git push origin --force --all
    git push origin --force --tags

Then, in order:
  1. Confirm the old ElevenLabs key is revoked (not just replaced).
  2. Tell anyone with a clone to delete it and clone again.
  3. On GitHub, check Insights > Forks. A fork keeps the old history and this
     rewrite cannot reach it.
EOF
