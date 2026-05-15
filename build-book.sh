#!/bin/bash

# Build script for converting markdown to PDF using eisvogel template
# Based on: https://github.com/Wandmalfarbe/pandoc-latex-template

set -e

INPUT_FILE="README.md"
OUTPUT_FILE="smol-training-zh.pdf"
TEMPLATE="eisvogel.latex"

echo "Building PDF from ${INPUT_FILE} using eisvogel template..."

# Check if template exists
if [ ! -f "${TEMPLATE}" ]; then
    echo "Error: ${TEMPLATE} not found!"
    echo "Download from: https://github.com/Wandmalfarbe/pandoc-latex-template/releases"
    exit 1
fi

# Add YAML metadata header for book mode
TEMP_FILE=$(mktemp /tmp/smol-training-XXXXXX.md 2>/dev/null || mktemp /tmp/smol-training.XXXXXX.md)
cat > "${TEMP_FILE}" << 'METADATA'
---
title: "Smol 训练手册：打造世界级 LLM 的秘诀"
author: ["HuggingFace Team"]
date: "2025-10-30"
subject: "LLM Training Guide"
keywords: [LLM, Training, Machine Learning, AI]
book: true
classoption: [oneside, openany]
documentclass: report
geometry: margin=1in
fontsize: 11pt
toc: true
toc-depth: 3
number-sections: true
colorlinks: true
linkcolor: blue
urlcolor: blue
toccolor: gray
header-includes:
  - \usepackage{xeCJK}
  - \setCJKmainfont{Songti SC}
  - \setCJKsansfont{PingFang SC}
  - \setCJKmonofont{Menlo}
  - \setmainfont{Songti SC}
  - \setsansfont{PingFang SC}
  - \setmonofont{Menlo}
---

METADATA

# Append the original content (skip the first few lines which are just title/links)
# Remove problematic webp images that can't be processed
tail -n +6 "${INPUT_FILE}" | sed 's/!\[.*\](.*\.webp)//g' >> "${TEMP_FILE}"

echo "Building PDF..."

pandoc "${TEMP_FILE}" \
  --from markdown+tex_math_dollars \
  --to pdf \
  --template "${TEMPLATE}" \
  --pdf-engine=xelatex \
  --syntax-highlighting=tango \
  --top-level-division=chapter \
  --wrap=auto \
  -o "${OUTPUT_FILE}"

# Clean up
rm -f "${TEMP_FILE}"

echo "PDF generated: ${OUTPUT_FILE}"
echo "Done!"
