#!/usr/bin/env bash
# Start one persistent preparation job from the already-installed environment.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .runtime
if ! command -v flock >/dev/null; then
  printf '%s\n' 'flock is required to prevent duplicate preparation jobs.' >&2
  exit 1
fi
if ! flock -n .runtime/prepare.lock true; then
  printf '%s\n' 'Preparation is already running. Read .runtime/prepare-resume.log'
  exit 0
fi
export DIG_START_STEP=3
export DIG_MODEL_TRANSPORT=curl
export DIG_MODEL_SOURCE="${DIG_MODEL_SOURCE:-auto}"
export DIG_MODEL_CONNECTIONS="${DIG_MODEL_CONNECTIONS:-4}"
export DIG_DOWNLOAD_TRANSPORT=curl
log=".runtime/prepare-resume-$(date +%Y%m%d-%H%M%S)-$$.log"
ln -sfn "$(basename "$log")" .runtime/prepare-resume.log
nohup bash scripts/prepare_cpu_online.sh >"$log" 2>&1 < /dev/null &
printf 'Preparation launched (PID %s). Existing weight prefixes will be reused.\n' "$!"
printf '%s\n' 'Progress: tail -f .runtime/prepare-resume.log' 'Success requires CPU_READY at the end of the log.'
