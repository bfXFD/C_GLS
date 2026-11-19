#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/Replica_V2}"
ARCHIVE="${ARCHIVE:-$ROOT/data/vmap.zip}"
URL="${URL:-https://huggingface.co/datasets/kxic/vMAP/resolve/main/vmap.zip}"

mkdir -p "$(dirname "$ARCHIVE")" "$DATA_ROOT"
if [[ ! -f "$ARCHIVE" ]]; then
  wget -O "$ARCHIVE" "$URL"
fi
tmp="$(mktemp -d "$ROOT/data/replicav2_extract.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT
unzip -q "$ARCHIVE" -d "$tmp"
source_root="$tmp/vmap"
[[ -d "$source_root" ]] || source_root="$tmp"
for scene in room_0 room_1 room_2 office_0 office_1 office_2 office_3 office_4; do
  candidate="$source_root/$scene"
  [[ -d "$candidate" ]] || continue
  rm -rf "$DATA_ROOT/$scene"
  cp -a "$candidate" "$DATA_ROOT/$scene"
done
python "$ROOT/scripts/fsm_replicav2_preflight.py" --data-root "$DATA_ROOT" --scenes room_0 office_0
