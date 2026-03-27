#!/usr/bin/env bash
# Push run_video_eval_suite.py to RunPod without scp/sftp (RunPod proxy often blocks those).
#
# Run this on your Mac from the ptlflow repo root:
#   chmod +x scripts/runpod_push_run_video_eval.sh
#   RUNPOD_USER='you@ssh.runpod.io' ./scripts/runpod_push_run_video_eval.sh
#
# Optional env:
#   RUNPOD_SSH_KEY   default: ~/.ssh/id_ed25519
#   RUNPOD_REMOTE_DIR  default: ~/ptlflow  (must exist or mkdir -p below)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL="$ROOT/run_video_eval_suite.py"

if [[ ! -f "$LOCAL" ]]; then
  echo "Missing $LOCAL" >&2
  exit 1
fi

: "${RUNPOD_USER:?Set RUNPOD_USER, e.g. jxx1c1alumvcja-64411927@ssh.runpod.io}"
KEY="${RUNPOD_SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_DIR="${RUNPOD_REMOTE_DIR:-~/ptlflow}"

echo "Uploading to ${RUNPOD_USER}:${REMOTE_DIR}/run_video_eval_suite.py"
ssh -tt -i "$KEY" "$RUNPOD_USER" "mkdir -p $REMOTE_DIR && cat > $REMOTE_DIR/run_video_eval_suite.py" < "$LOCAL"
echo "Done."
