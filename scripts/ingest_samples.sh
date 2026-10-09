#!/usr/bin/env bash
# Upload every file in ./samples to a running API (default http://localhost:8000).
set -euo pipefail
API="${API_URL:-http://localhost:8000}"
for f in "$(dirname "$0")"/../samples/*; do
  echo "-> $(basename "$f")"
  curl -sS -F "file=@${f}" "${API}/documents"
  echo
done
