#!/usr/bin/env bash
# Publish the M2 SFT adapter (models/m2/sft_adapter) as the private Kaggle dataset
# `align-pareto-m2-adapters`, used by the M3/M4 notebooks and by Group B (docs/EVAL_PROTOCOL.md §4).
# Creates the dataset the first time, then adds a new version. Usage: kaggle/publish_adapters.sh
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
KAGGLE=${KAGGLE:-$ROOT/.venv/bin/kaggle}
DATASET=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['id'])" "$ROOT/kaggle/m2-adapters/dataset-metadata.json")
[[ -f $ROOT/models/m2/sft_adapter/adapter_config.json ]] || { echo "models/m2/sft_adapter missing: run M2 first"; exit 1; }
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/models/m2"
cp "$ROOT/kaggle/m2-adapters/dataset-metadata.json" "$STAGE/"
cp -r "$ROOT/models/m2/sft_adapter" "$STAGE/models/m2/"
cp "$ROOT/results/m2/metrics.json" "$ROOT/results/m2/experiment_config.json" "$STAGE/" 2>/dev/null || true

MSG="M2 SFT adapter from $(git -C "$ROOT" rev-parse --short HEAD)"
if "$KAGGLE" datasets files "$DATASET" >/dev/null 2>&1; then
    "$KAGGLE" datasets version -p "$STAGE" -m "$MSG" -r zip
else
    "$KAGGLE" datasets create -p "$STAGE" -r zip   # private by default
fi
echo "Published $DATASET. Share it with Group B: https://www.kaggle.com/datasets/$DATASET -> Settings -> Collaborators"
