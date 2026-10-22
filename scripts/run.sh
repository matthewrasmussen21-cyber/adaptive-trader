#!/usr/bin/env bash
# Run a command; on failure, surface the tail of its output as a GitHub error annotation
# (annotations are visible through the API even where raw logs are not).
set -o pipefail
title="$1"; shift
log=$(mktemp)
"$@" 2>&1 | tee "$log"
code=${PIPESTATUS[0]}
if [ "$code" -ne 0 ]; then
  tail -n 60 "$log" | sed 's/%/%25/g' | awk 'BEGIN{ORS="%0A"} {print}' > "$log.tail"
  echo "::error title=${title} failed (exit ${code})::$(cat "$log.tail")"
fi
exit "$code"
