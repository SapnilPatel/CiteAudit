#!/usr/bin/env bash
set -euo pipefail

mkdir -p upstream

clone_or_update() {
  local url="$1"
  local dir="$2"
  if [ -d "$dir/.git" ]; then
    echo "[update] $dir"
    git -C "$dir" pull --ff-only
  else
    echo "[clone] $url -> $dir"
    git clone "$url" "$dir"
  fi
}

clone_or_update "https://github.com/ScienceNLP-Lab/Citation-Integrity.git" "upstream/Citation-Integrity"
clone_or_update "https://github.com/dwadden/multivers.git" "upstream/multivers"

echo
echo "Done. Next: python scripts/locate_upstream.py"
