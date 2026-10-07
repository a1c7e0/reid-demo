#!/usr/bin/env bash
# Local sanity check: build, start, import ./import images, verify persons.
# Deliberately tiny (a handful of images) — not a benchmark.
set -euo pipefail

PORT="${PORT:-8010}"
BASE="http://127.0.0.1:${PORT}"
DIR="$(cd "$(dirname "$0")/.." && pwd)"

cd "$DIR"
mkdir -p import data

echo "[smoke] building image (cached after first run)"
docker build -t reid-gallery:smoke .

echo "[smoke] starting container on :${PORT}"
docker rm -f reid-gallery-smoke >/dev/null 2>&1 || true
docker run -d --name reid-gallery-smoke -p "127.0.0.1:${PORT}:8000" \
  -v "$DIR/import:/import:ro" -v "$DIR/data:/data" reid-gallery:smoke

cleanup() { docker rm -f reid-gallery-smoke >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "[smoke] waiting for API"
for _ in $(seq 1 60); do
  if curl -fsS "$BASE/api/status" >/dev/null 2>&1; then break; fi
  sleep 1
done

echo "[smoke] scanning /import"
curl -fsS -X POST "$BASE/api/scan" -H 'Content-Type: application/json' \
  -d '{"path": "/import"}'

echo "[smoke] waiting for analysis"
for _ in $(seq 1 120); do
  pending=$(curl -fsS "$BASE/api/status" | python3 -c 'import sys,json;print(json.load(sys.stdin)["pending"])')
  [ "$pending" = "0" ] && break
  sleep 1
done

echo "[smoke] status:"
curl -fsS "$BASE/api/status"
echo
echo "[smoke] persons:"
curl -fsS "$BASE/api/persons"
echo
echo "[smoke] OK"
