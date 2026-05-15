#!/bin/bash

# Build script using eisvogel template with proper configuration

set -e

INPUT_FILE="README.md"
OUTPUT_FILE="smol-training-zh.pdf"

echo "Building PDF from ${INPUT_FILE}..."

pandoc "${INPUT_FILE}" \
  --from markdown \
  --to pdf \
  --template eisvogel \
  --pdf-engine=xelatex \
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
  -V titlepage=true \
  -V titlepage-color="2E86C1" \
  -V titlepage-text-color="FFFFFF" \
  -V toc-own-page=true \
  --toc \
  --toc-depth=3 \
  --number-sections \
  --syntax-highlighting=tango \
  -o "${OUTPUT_FILE}"

echo "PDF generated: ${OUTPUT_FILE}"
