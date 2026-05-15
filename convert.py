#!/usr/bin/env python3

"""
Convert markdown to PDF using pandoc and xelatex
"""

import subprocess
import sys

def convert_markdown_to_pdf(input_file, output_file):
    """Convert markdown file to PDF"""

    cmd = [
        'pandoc',
        input_file,
        '--from', 'markdown',
        '--to', 'pdf',
        '--pdf-engine=xelatex',
        '-V', 'documentclass=report',
        '-V', 'geometry:margin=1in',
        '-V', 'fontsize=11pt',
        '-V', 'CJKmainfont=Songti SC',
        '-V', 'mainfont=Songti SC',
        '-V', 'sansfont=Hiragino Sans GB',
        '-V', 'monofont=Menlo',
        '-V', 'colorlinks=true',
        '-V', 'linkcolor=blue',
        '-V', 'urlcolor=blue',
        '-V', 'toccolor=gray',
        '--toc',
        '--toc-depth=3',
        '--number-sections',
        '--syntax-highlighting=tango',
        '-o', output_file
    ]

    print(f"Converting {input_file} to {output_file}...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: {result.stderr}", file=sys.stderr)
        return False

    print(f"Successfully generated {output_file}")
    return True

if __name__ == '__main__':
    input_file = 'README.md'
    output_file = 'smol-training-zh.pdf'

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    success = convert_markdown_to_pdf(input_file, output_file)
    sys.exit(0 if success else 1)
