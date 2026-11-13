#!/usr/bin/env bash
# Push a Kaggle kernel pinned to a pushed commit of this repo.
#   scripts/kaggle_push.sh kaggle/train_eval [git-ref]
# Requires the kaggle CLI to be authenticated (KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle).
set -euo pipefail
src=${1:?kernel dir}
ref=${2:-$(git rev-parse HEAD)}
build="kaggle/build/$(basename "$src")"
rm -rf "$build" && mkdir -p "$build"
cp "$src"/* "$build"/
sed -i "s/__REPO_REF__/$ref/g" "$build"/*.py
kaggle kernels push -p "$build"
