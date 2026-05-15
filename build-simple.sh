#!/bin/bash

# Simple build script using pandoc without eisvogel template

set -e

INPUT_FILE="README.md"
OUTPUT_FILE="smol-training-zh.pdf"

echo "Building PDF from ${INPUT_FILE}..."

pandoc "${INPUT_FILE}" \
  --from markdown \
  --to pdf \
  --pdf-engine=tectonic \
  -V documentclass=report \
  -V geometry:margin=1in \
  -V fontsize=11pt \
  -V CJKmainfont="Songti SC" \
  -V mainfont="Songti SC" \
  -V sansfont="Hiragino Sans GB" \
  -V monofont="Menlo" \
  -V colorlinks=true \
  -V linkcolor=blue \
  -V urlcolor=blue \
  -V toccolor=gray \
  --toc \
  --toc-depth=3 \
  --number-sections \
  --syntax-highlighting=tango \
  -o "${OUTPUT_FILE}"

echo "PDF generated: ${OUTPUT_FILE}"
