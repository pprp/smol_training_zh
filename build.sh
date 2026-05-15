#!/bin/bash

# Build script for converting markdown to PDF using pandoc + eisvogel template

set -e

INPUT_FILE="README.md"
OUTPUT_FILE="smol-training-zh.pdf"

echo "Building PDF from ${INPUT_FILE}..."

pandoc "${INPUT_FILE}" \
  --from markdown \
  --to pdf \
  --template eisvogel \
  --pdf-engine=xelatex \
  -V documentclass=book \
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
  --highlight-style=tango \
  -o "${OUTPUT_FILE}"

echo "PDF generated: ${OUTPUT_FILE}"
