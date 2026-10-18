#!/bin/bash
# Builds release/influence-{linux,windows}-x64 archives in Docker. Usage: scripts/release/make_release.sh [model file in models/run1]
set -euo pipefail
cd "$(dirname "$0")/../.."
MODEL=${1:-$(ls -t models/run1/*.bin.gz | head -1 | xargs basename)}
WORK=${WORK:-$PWD/release/work}
mkdir -p release "$WORK"
docker build -q -t influence-release scripts/release >/dev/null
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/work -e CARGO_HOME=/work/cargo-home -e MODEL="$MODEL" \
  -v "$PWD":/src:ro -v "$WORK":/work -v "$PWD/release":/out influence-release bash /src/scripts/release/build.sh
