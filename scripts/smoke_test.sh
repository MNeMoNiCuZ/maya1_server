#!/usr/bin/env bash
#
# End-to-end smoke test against a running Maya1 server.
# Checks /health, lists emotions, then synthesises a short clip to out.wav.
#
# Usage:
#   scripts/smoke_test.sh                     # targets http://localhost:41217
#   MAYA1_BASE_URL=http://host:41217 scripts/smoke_test.sh
set -euo pipefail

BASE_URL="${MAYA1_BASE_URL:-http://localhost:41217}"
OUT_FILE="${1:-out.wav}"

echo "== 1. Health =="
curl -fsS "${BASE_URL}/health"
echo

echo "== 2. Supported emotions =="
curl -fsS "${BASE_URL}/v1/emotions"
echo

echo "== 3. Synthesise -> ${OUT_FILE} =="
curl -fsS -X POST "${BASE_URL}/v1/audio/speech" \
    -H "Content-Type: application/json" \
    -d '{
          "description": "Female, in her 30s with an American accent, warm and energetic, clear diction.",
          "input": "Hey there! This is Maya1 running in Docker <laugh> and it sounds amazing.",
          "response_format": "wav"
        }' \
    --output "${OUT_FILE}" \
    --dump-header /tmp/maya1_headers.txt

echo "Saved audio to ${OUT_FILE}"
grep -i "X-Audio-Duration-Seconds\|X-Generated-Tokens\|X-Elapsed-Seconds" /tmp/maya1_headers.txt || true
echo "Smoke test complete."
