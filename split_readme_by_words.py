#!/usr/bin/env python3
"""
Script to split README_new.md into four parts with approximately equal word counts.
"""

import os

def count_words(text):
    """Count the number of words in a text."""
    return len(text.split())

def split_markdown_by_words(file_path, num_parts=4):
    """
    Split a markdown file into parts with approximately equal word counts,
    preserving line structure.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Calculate total words
    total_words = sum(len(line.split()) for line in lines)
    words_per_part = total_words // num_parts

    print(f"Total words: {total_words}")
    print(f"Words per part: {words_per_part}")

    parts = []
    current_part = []
    current_word_count = 0

    for line in lines:
        line_words = len(line.split())
        if current_word_count + line_words > words_per_part and len(parts) < num_parts - 1:
            # Start a new part
            parts.append(''.join(current_part))
            current_part = [line]
            current_word_count = line_words
        else:
            current_part.append(line)
            current_word_count += line_words

    # Add the last part
    if current_part:
        parts.append(''.join(current_part))

    return parts

def main():
    file_path = 'README_new.md'
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return

    parts = split_markdown_by_words(file_path, 4)

    for i, part in enumerate(parts, 1):
        output_file = f'part{i}.md'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(part)
        word_count = count_words(part)
        print(f'Part {i} written to {output_file} with {word_count} words')

if __name__ == '__main__':
    main()