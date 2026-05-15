#!/bin/bash

# Build script using pandoc -> HTML -> weasyprint

set -e

INPUT_FILE="README.md"
HTML_FILE="smol-training-zh.html"
OUTPUT_FILE="smol-training-zh.pdf"

echo "Converting markdown to HTML..."
pandoc "${INPUT_FILE}" \
  --from markdown \
  --to html5 \
  --standalone \
  --toc \
  --toc-depth=3 \
  --number-sections \
  --metadata title="Smol 训练手册：打造世界级 LLM 的秘诀 (中文版)" \
  -o "${HTML_FILE}"

echo "Converting HTML to PDF with weasyprint..."
weasyprint "${HTML_FILE}" "${OUTPUT_FILE}"

echo "PDF generated: ${OUTPUT_FILE}"
