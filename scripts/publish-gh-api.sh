#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Sync tracked files to GitHub via Contents API (no git push).

Usage:
  scripts/publish-gh-api.sh [options]

Options:
  --owner <org_or_user>   GitHub owner (default: $GITHUB_OWNER or "stouffer-labs")
  --repo <name>           Repository name (default: current directory name)
  --branch <name>         Branch to write (default: main)
  --create-repo           Create the GitHub repo if it does not exist
  --public                Visibility when creating repo (default)
  --private               Visibility when creating repo
  --dry-run               Show actions without writing
  -h, --help              Show this help

Notes:
  - Requires: gh CLI authenticated with repo scope.
  - Sync source is `git ls-files` when in a git repo.
  - If no git repo is present, it falls back to filesystem enumeration with built-in excludes.
  - Each file becomes a separate commit in GitHub.
  - This command adds/updates tracked files; it does not delete remote-only files.
EOF
}

OWNER="${GITHUB_OWNER:-stouffer-labs}"
REPO="$(basename "$(pwd)")"
BRANCH="main"
DRY_RUN=0
CREATE_REPO=0
VISIBILITY="public"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --owner)
      OWNER="${2:-}"
      shift 2
      ;;
    --repo)
      REPO="${2:-}"
      shift 2
      ;;
    --branch)
      BRANCH="${2:-}"
      shift 2
      ;;
    --create-repo)
      CREATE_REPO=1
      shift
      ;;
    --public)
      VISIBILITY="public"
      shift
      ;;
    --private)
      VISIBILITY="private"
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v gh >/dev/null 2>&1; then
  echo "error: gh CLI not found" >&2
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "error: gh is not authenticated; run: gh auth login" >&2
  exit 1
fi

if ! gh api "repos/${OWNER}/${REPO}" >/dev/null 2>&1; then
  if [[ "$CREATE_REPO" -eq 1 ]]; then
    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "dry-run: would create repo ${OWNER}/${REPO} (${VISIBILITY})"
    else
      echo "repo not found: creating ${OWNER}/${REPO} (${VISIBILITY})..."
      gh repo create "${OWNER}/${REPO}" "--${VISIBILITY}" >/dev/null
      echo "repo created: ${OWNER}/${REPO}"
    fi
  else
    echo "error: repo '${OWNER}/${REPO}' not found or not accessible (HTTP 404)." >&2
    echo "hint: create it first, or rerun with --create-repo." >&2
    echo "      gh repo create ${OWNER}/${REPO} --public" >&2
    exit 1
  fi
fi

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  mapfile -t FILES < <(git ls-files)
  SOURCE_MODE="git"
else
  # No git repo: enumerate files directly with conservative excludes.
  mapfile -t FILES < <(
    find . \
      \( -path './.cypress' -o -path './.cypress/*' \
      -o -path './target' -o -path './target/*' \
      -o -path './internal' -o -path './internal/*' \
      -o -path './.tmp' -o -path './.tmp/*' \
      -o -path './dist' -o -path './dist/*' \
      -o -path './build' -o -path './build/*' \
      -o -path './venv' -o -path './venv/*' \
      -o -path './.venv' -o -path './.venv/*' \
      -o -path './.git' -o -path './.git/*' \
      -o -path './node_modules' -o -path './node_modules/*' \
      -o -path './__pycache__' -o -path './__pycache__/*' \) -prune \
      -o -type f ! -name '.DS_Store' -print \
    | sed 's#^\./##'
  )
  SOURCE_MODE="filesystem"
fi

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "nothing to sync (no files found in ${SOURCE_MODE} mode)"
  exit 0
fi

echo "sync target: ${OWNER}/${REPO} (branch=${BRANCH})"
echo "source mode: ${SOURCE_MODE}"
echo "files: ${#FILES[@]}"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "mode: dry-run"
fi

added=0
updated=0
skipped=0
failed=0

for file in "${FILES[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "skip: $file (not a regular file)"
    skipped=$((skipped + 1))
    continue
  fi

  sha="$(gh api "repos/${OWNER}/${REPO}/contents/${file}" --jq '.sha' 2>/dev/null || true)"
  content="$(base64 -i "$file" | tr -d '\n')"
  payload="$(mktemp -t cypress-publish-payload.XXXXXX)"

  if [[ -n "$sha" ]]; then
    action="update"
    message="Update ${file}"
    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "dry-run: ${action} ${file}"
      updated=$((updated + 1))
      rm -f "$payload"
      continue
    fi
    jq -n \
      --arg message "$message" \
      --arg content "$content" \
      --arg sha "$sha" \
      --arg branch "$BRANCH" \
      '{message:$message, content:$content, sha:$sha, branch:$branch}' >"$payload"
    if gh api --method PUT "repos/${OWNER}/${REPO}/contents/${file}" --input "$payload" >/dev/null; then
      echo "updated: $file"
      updated=$((updated + 1))
    else
      echo "failed: $file" >&2
      failed=$((failed + 1))
    fi
  else
    action="add"
    message="Add ${file}"
    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "dry-run: ${action} ${file}"
      added=$((added + 1))
      rm -f "$payload"
      continue
    fi
    jq -n \
      --arg message "$message" \
      --arg content "$content" \
      --arg branch "$BRANCH" \
      '{message:$message, content:$content, branch:$branch}' >"$payload"
    if gh api --method PUT "repos/${OWNER}/${REPO}/contents/${file}" --input "$payload" >/dev/null; then
      echo "added: $file"
      added=$((added + 1))
    else
      echo "failed: $file" >&2
      failed=$((failed + 1))
    fi
  fi
  rm -f "$payload"
done

echo
echo "summary: added=${added} updated=${updated} skipped=${skipped} failed=${failed}"
if [[ "$failed" -gt 0 ]]; then
  exit 1
fi
