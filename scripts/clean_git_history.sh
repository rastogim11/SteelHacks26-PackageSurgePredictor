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
# ROTATE THE ELEVENLABS KEY. This script is the cleanup, not the fix.
#
# Safe to run more than once.
#
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

# Matched at ANY path. The first run of this script missed
# packagestats_cleaned.xlsx because the two oldest commits kept it at the repo
# root and only later commits moved it under files/ - an exact path list cannot
# see a file that moved, so match on the name instead.
GLOBS_TO_PURGE=(
  "*packagestats*"
  "*searchresults*"
)
PATHS_TO_PURGE=(
  "elevenLabs.py"   # held a live ElevenLabs API key
)

# Real files to carry across the rewrite. filter-repo ends with a hard reset
# onto the rewritten HEAD, which deletes purged paths from the WORKING TREE as
# well as from history, and packagestats.csv is the pipeline's only input.
DATA_FILES=(
  "packagestats.csv"
  "packagestats.xlsx"
  "files/packagestats_cleaned.xlsx"
  "searchresults(1).csv"
)

echo "Repository: $REPO_ROOT"
echo
echo "Removed from ALL commits, at any path:"
printf '  %s\n' "${GLOBS_TO_PURGE[@]}" "${PATHS_TO_PURGE[@]}"
echo
echo "Every commit SHA changes. The remote is overwritten with --force."
echo "Anyone else with a clone must re-clone afterwards."
echo
read -r -p "Type REWRITE to continue: " confirm
[ "$confirm" = "REWRITE" ] || { echo "Aborted."; exit 1; }

# git-filter-repo is not part of git; it is installed in .venv here.
if [ -x "$REPO_ROOT/.venv/bin/git-filter-repo" ]; then
  PATH="$REPO_ROOT/.venv/bin:$PATH"
  export PATH
fi
if ! command -v git-filter-repo >/dev/null 2>&1; then
  echo
  echo "git-filter-repo is not installed. Install it with:"
  echo "    .venv/bin/python -m pip install git-filter-repo"
  exit 1
fi

STAMP="$(date +%Y%m%d-%H%M%S)"

BACKUP="../SteelHacks26-backup-$STAMP.git"
echo
echo "Backing up history to $BACKUP"
git clone --mirror . "$BACKUP"

DATA_BACKUP="../SteelHacks26-data-$STAMP"
echo "Preserving raw data to $DATA_BACKUP"
mkdir -p "$DATA_BACKUP/files"
for p in "${DATA_FILES[@]}"; do
  [ -f "$p" ] || continue
  cp -p "$p" "$DATA_BACKUP/$p"
  echo "  saved $p"
done

REMOTE_URL="$(git remote get-url origin 2>/dev/null || echo '')"

# A stash is a ref, so it keeps purged blobs reachable. Its only content here
# is an edit to a file being purged, and the working copy of that file is
# already saved above, so dropping it loses nothing.
if git rev-parse --verify --quiet refs/stash >/dev/null; then
  echo
  echo "Dropping stash (it references purged blobs; contents already saved)"
  git stash clear
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Discarding working-tree edits to purged files (copies are in $DATA_BACKUP)"
  git checkout -- . 2>/dev/null || true
fi

echo "Rewriting history..."
ARGS=()
for g in "${GLOBS_TO_PURGE[@]}"; do ARGS+=(--path-glob "$g"); done
for p in "${PATHS_TO_PURGE[@]}"; do ARGS+=(--path "$p"); done
git filter-repo --invert-paths "${ARGS[@]}" --force

# filter-repo drops the remote on purpose, so a force push cannot happen by
# reflex. Putting it back is the deliberate step.
if [ -n "$REMOTE_URL" ]; then
  git remote add origin "$REMOTE_URL" 2>/dev/null \
    || git remote set-url origin "$REMOTE_URL"
fi

echo
echo "Restoring raw data from $DATA_BACKUP"
for p in "${DATA_FILES[@]}"; do
  [ -f "$DATA_BACKUP/$p" ] || continue
  mkdir -p "$(dirname "$p")"
  cp -p "$DATA_BACKUP/$p" "$p"
  echo "  restored $p"
done

echo
echo "History rewritten. Local checks:"
echo -n "  paths still tracked:    "
git ls-files | grep -Ei 'packagestats|searchresults|elevenLabs' || echo "none"
echo -n "  blobs still in history: "
git rev-list --objects --all \
  | grep -Ei 'packagestats|searchresults|elevenLabs' || echo "none"
echo -n "  refs present:           "
git for-each-ref --format='%(refname)' | tr '\n' ' '; echo
echo -n "  pipeline input:         "
if [ -f packagestats.csv ]; then
  echo "packagestats.csv OK"
else
  echo "MISSING - copy it back from $DATA_BACKUP"
fi
echo -n "  remote:                 "
git remote get-url origin 2>/dev/null || echo "NOT SET"

cat <<'EOF'

Both check lines above must read "none" before you push.

To publish the rewrite:

    git push origin --force --all

Then:
  1. Confirm the old ElevenLabs key is revoked, not just replaced.
  2. Tell anyone with a clone to delete it and clone again.
  3. A force push can leave orphaned commits reachable on GitHub by direct SHA.
     To be certain the names are gone, delete the repository on GitHub and
     recreate it, or ask GitHub Support to garbage-collect.
EOF
