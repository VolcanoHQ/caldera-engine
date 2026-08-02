#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <input-manuscript> [output-manifest]" >&2
  echo "Example: $0 data/corpus/MyNewBook.txt scratch/mybook_manifest.json" >&2
  exit 1
fi

INPUT_MANUSCRIPT="$1"
OUTPUT_MANIFEST="${2:-scratch/$(basename "${INPUT_MANUSCRIPT%.*}")_manifest.json}"

if [[ ! -f "$INPUT_MANUSCRIPT" ]]; then
  echo "Input manuscript not found: $INPUT_MANUSCRIPT" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_MANIFEST")"

echo "Analyzing manuscript: $INPUT_MANUSCRIPT"
python -m src.tier_1_parser --input "$INPUT_MANUSCRIPT" --output "$OUTPUT_MANIFEST"

echo
echo "Manifest written to: $OUTPUT_MANIFEST"
echo
echo "Start the local app with:"
echo "  python -m src.gui_server"
echo "Then open http://localhost:8082/"
