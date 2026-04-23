#!/usr/bin/env bash

set -euo pipefail

LIVE_URL="https://cdn-resources.ableton.com/resources/pdfs/live-manual/12/2026-03-20/live12-manual-en.pdf"
PUSH_URL="https://cdn-resources.ableton.com/resources/pdfs/push-manual/3/2025-09-17/push3-manual-en.pdf"
MUSIC_STRATEGIES_URL="https://cdn-resources.ableton.com/resources/uploads/makingmusic/MakingMusic_DennisDeSantis.pdf"

OUTPUT_DIR="${1:-corpus}"
mkdir -p "$OUTPUT_DIR"

download() {
    local url="$1"
    local path="$2"

    if [ -f "$path" ]; then
        echo "File already exists: $path"
        return 0
    fi

    echo "Downloading $path..."
    curl -fSL --retry 3 --retry-delay 5 -o "$path" "$url"
    echo "Saved to $path"
}

download "$LIVE_URL" "$OUTPUT_DIR/live_12.pdf"
download "$PUSH_URL" "$OUTPUT_DIR/push_3.pdf"
download "$MUSIC_STRATEGIES_URL" "$OUTPUT_DIR/making_music.pdf"
