#!/usr/bin/env bash
# Run one milestone notebook on Kaggle from this machine and bring its outputs back.
#   1. upload src/, configs/, data/eval/, requirements.txt as a new version of the dataset `align-pareto-src`
#   2. push notebooks/<code_file> as the kernel in kaggle/<milestone>/kernel-metadata.json
#   3. wait for the run, download its output to outputs/kaggle/<milestone>/ and copy
#      results/<milestone>, models/<milestone>, data/processed/<milestone> into the repo
# Usage: kaggle/run_on_kaggle.sh m1 [--skip-upload]
set -euo pipefail

MILESTONE=${1:?usage: kaggle/run_on_kaggle.sh <m1|m2|...> [--skip-upload]}
SKIP_UPLOAD=${2:-}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
KAGGLE=${KAGGLE:-$ROOT/.venv/bin/kaggle}
META="$ROOT/kaggle/$MILESTONE/kernel-metadata.json"
[[ -f $META ]] || { echo "No $META"; exit 1; }
KERNEL=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['id'])" "$META")
NOTEBOOK=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['code_file'])" "$META")
SRC_DATASET=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['id'])" "$ROOT/kaggle/src-dataset/dataset-metadata.json")
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
log() { echo "[$(date +%H:%M:%S)] $*"; }

# 1. Code dataset
if [[ $SKIP_UPLOAD != --skip-upload ]]; then
    mkdir -p "$STAGE/src-dataset"
    cp "$ROOT/kaggle/src-dataset/dataset-metadata.json" "$ROOT/requirements.txt" "$STAGE/src-dataset/"
    cp -r "$ROOT/src" "$ROOT/configs" "$STAGE/src-dataset/"
    mkdir -p "$STAGE/src-dataset/data" && cp -r "$ROOT/data/eval" "$STAGE/src-dataset/data/"
    find "$STAGE/src-dataset" -name __pycache__ -prune -exec rm -rf {} +
    find "$STAGE/src-dataset" -name '.~lock*' -delete   # LibreOffice lock files break the upload
    VERSION_MSG="$(git -C "$ROOT" rev-parse --short HEAD)$(git -C "$ROOT" diff --quiet HEAD -- src configs data/eval requirements.txt || echo '+local-changes')"
    log "Uploading $SRC_DATASET ($VERSION_MSG)"
    UPLOAD_START=$(date -u +"%Y-%m-%d %H:%M:%S")
    "$KAGGLE" datasets version -p "$STAGE/src-dataset" -m "$VERSION_MSG" -r zip
    # `datasets status` reports the previous version as ready while the new one is processing, so wait until
    # the listed files carry a creation time after this upload
    for _ in $(seq 40); do
        NEWEST=$("$KAGGLE" datasets files "$SRC_DATASET" --page-size 200 2>/dev/null | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9:]{8}' | sort | tail -1)
        [[ -n $NEWEST && $NEWEST > $UPLOAD_START ]] && break
        log "Waiting for the new $SRC_DATASET version (latest files from ${NEWEST:-?})"; sleep 15
    done
    [[ -n $NEWEST && $NEWEST > $UPLOAD_START ]] || { echo "New $SRC_DATASET version did not appear; check https://www.kaggle.com/datasets/$SRC_DATASET"; exit 1; }
    log "New $SRC_DATASET version is live"
fi

# 2. Kernel
mkdir -p "$STAGE/kernel"
cp "$META" "$STAGE/kernel/"
cp "$ROOT/notebooks/$NOTEBOOK" "$STAGE/kernel/"
log "Pushing $KERNEL"
"$KAGGLE" kernels push -p "$STAGE/kernel"
log "Run page: https://www.kaggle.com/code/$KERNEL"

# 3. Wait, then download
sleep 30
while true; do
    STATUS=$("$KAGGLE" kernels status "$KERNEL" 2>&1 | tr '[:upper:]' '[:lower:]')
    case $STATUS in
        *complete*) log "Run complete"; break ;;
        *error*|*cancel*) log "Run failed: $STATUS"; FAILED=1; break ;;
        *) log "$STATUS"; sleep 60 ;;
    esac
done

OUT="$ROOT/outputs/kaggle/$MILESTONE"
rm -rf "$OUT"; mkdir -p "$OUT"
"$KAGGLE" kernels output "$KERNEL" -p "$OUT" -o
log "Kaggle output saved to $OUT"
[[ -z ${FAILED:-} ]] || { echo "See the log in $OUT"; exit 1; }

PROJECT="$OUT/align-pareto"
for sub in results models data/processed; do
    if [[ -d $PROJECT/$sub/$MILESTONE ]]; then
        rm -rf "${ROOT:?}/$sub/$MILESTONE"
        mkdir -p "$ROOT/$sub"
        cp -r "$PROJECT/$sub/$MILESTONE" "$ROOT/$sub/$MILESTONE"
        log "Updated $sub/$MILESTONE"
    fi
done
