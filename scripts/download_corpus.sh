#!/usr/bin/env bash

set -euo pipefail

URL="https://cdn-resources.ableton.com/resources/pdfs/live-manual/12/2026-03-20/live12-manual-en.pdf"
OUTPUT="${1:-corpus.pdf}"

if [ -f "$OUTPUT" ]; then
    echo "File already exists: $OUTPUT"
    exit 0
fi

echo "Downloading Ableton Live 12 manual..."
curl -fSL --retry 3 --retry-delay 5 -o "$OUTPUT" "$URL"
echo "Saved to $OUTPUT"
