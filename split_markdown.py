#!/usr/bin/env python3
"""
Script to split a large markdown file into semantic sections based on headings and content structure.
"""

import re
import os
from pathlib import Path

def split_markdown_into_sections(file_path, output_dir="sections"):
    """
    Split a markdown file into semantic sections based on major headings.

    Args:
        file_path (str): Path to the input markdown file
        output_dir (str): Directory to save the split sections
    """

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Read the entire file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split content into lines
    lines = content.split('\n')

    # Define major section markers based on the analysis
    major_sections = [
        ("introduction", "引言", 1),
        ("training-compass", "训练指南针", 36),
        ("why-question", "Why：没人愿意回答的问题", 51),
        ("what-decisions", "What：将目标转化为决策", 121),
        ("super-power", "超能力：速度与数据", 139),
        ("baseline-choice", "选择你的基线模型", 169),
        ("training-framework", "选择训练框架", 228),
        ("ablation-setup", "消融实验设置", 261),
        ("architecture-choices", "架构选择", 473),
        ("tokenizer", "分词器", 1184),
        ("optimiser-hyperparameters", "优化器与训练超参数", 1430),
        ("scaling-laws", "扩展定律", 1675),
        ("data-mixture", "数据混合与策划", 1707),
        ("pre-flight-checklist", "起飞前检查清单", 1821),
        ("scaling-surprises", "扩展中的意外", 1839),
        ("staying-course", "保持航向", 1987),
        ("mid-training", "中期训练", 2039),
        ("wrapping-pretraining", "预训练收尾", 2081),
        ("post-training-compass", "后训练指南针", 2108),
        ("evals-first", "首要之事：先搞评估", 2146),
        ("tools-trade", "行业工具", 2241),
        ("sft-start", "为何所有后训练流程都从SFT开始", 2277),
        ("preference-optimization", "从SFT到偏好优化", 2772),
        ("online-policy", "走向在线策略并超越监督标签", 2880),
        ("wrapping-post-training", "收尾：后训练阶段", 3074),
        ("gpu-architecture", "GPU内部：内部架构", 3097),
        ("gpu-communication", "GPU之外：GPU如何与外界通信", 3312),
        ("resilient-systems", "构建弹性训练系统", 4051),
        ("optimizing-throughput", "优化训练吞吐量", 4171),
        ("acknowledgments", "致谢", 4323)
    ]

    # Initialize sections
    sections = []

    # Find the actual line numbers for each section
    section_boundaries = []
    for i, (section_id, title, expected_line) in enumerate(major_sections):
        # Search for the section title in the file
        found_line = None
        for line_num, line in enumerate(lines, 1):
            if title in line and line.startswith('#'):
                found_line = line_num
                break

        if found_line:
            section_boundaries.append((section_id, title, found_line))
        else:
            print(f"Warning: Could not find section '{title}'")

    # Sort by line number
    section_boundaries.sort(key=lambda x: x[2])

    # Create sections by extracting content between boundaries
    for i, (section_id, title, start_line) in enumerate(section_boundaries):
        end_line = section_boundaries[i + 1][2] if i + 1 < len(section_boundaries) else len(lines)

        # Extract content for this section
        section_lines = lines[start_line - 1:end_line - 1]  # Convert to 0-indexed

        # Remove leading empty lines
        while section_lines and not section_lines[0].strip():
            section_lines.pop(0)

        # Remove trailing empty lines
        while section_lines and not section_lines[-1].strip():
            section_lines.pop()

        section_content = '\n'.join(section_lines)

        if section_content.strip():
            sections.append({
                'id': section_id,
                'title': title,
                'content': section_content,
                'start_line': start_line,
                'end_line': end_line - 1
            })

    # Write sections to files
    for section in sections:
        filename = f"{section['id']}.md"
        filepath = os.path.join(output_dir, filename)

        # Add section title as main heading if not present
        content = section['content']
        if not content.startswith('#'):
            content = f"# {section['title']}\n\n{content}"

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"Created: {filepath} (lines {section['start_line']}-{section['end_line']})")

    # Create an index file
    create_index_file(sections, output_dir)

    print(f"\nSplit {len(sections)} sections into '{output_dir}' directory")
    return sections

def create_index_file(sections, output_dir):
    """Create an index file with links to all sections."""

    index_content = "# Smol 训练手册 - 章节索引\n\n"
    index_content += "本文件是根据语义信息拆分后的章节索引。\n\n"
    index_content += "## 章节列表\n\n"

    for section in sections:
        index_content += f"- [{section['title']}]({section['id']}.md) (第 {section['start_line']}-{section['end_line']} 行)\n"

    index_filepath = os.path.join(output_dir, "README.md")
    with open(index_filepath, 'w', encoding='utf-8') as f:
        f.write(index_content)

    print(f"Created index: {index_filepath}")

def create_summary_sections(file_path, output_dir="sections"):
    """Create additional semantic groupings based on content analysis."""

    # Read the original file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Define thematic groupings
    thematic_sections = [
        {
            "id": "pretraining-complete",
            "title": "预训练完整指南",
            "description": "从零开始预训练一个LLM的完整流程",
            "keywords": ["预训练", "数据", "架构", "超参数", "消融实验"]
        },
        {
            "id": "post-training-complete",
            "title": "后训练完整指南",
            "description": "SFT、DPO、RLHF等后训练技术",
            "keywords": ["后训练", "SFT", "DPO", "偏好优化", "RL"]
        },
        {
            "id": "infrastructure-complete",
            "title": "基础设施完整指南",
            "description": "GPU集群、网络、存储和性能优化",
            "keywords": ["GPU", "基础设施", "并行", "网络", "存储"]
        }
    ]

    # Create summary files for each thematic section
    for theme in thematic_sections:
        theme_content = f"# {theme['title']}\n\n"
        theme_content += f"{theme['description']}\n\n"
        theme_content += "## 相关章节\n\n"

        # This is a simplified approach - in a real implementation,
        # you would analyze content to match keywords
        if "pretraining" in theme['id']:
            theme_content += "- [训练指南针](training-compass.md)\n"
            theme_content += "- [架构选择](architecture-choices.md)\n"
            theme_content += "- [数据混合与策划](data-mixture.md)\n"
            theme_content += "- [扩展定律](scaling-laws.md)\n"
        elif "post-training" in theme['id']:
            theme_content += "- [后训练指南针](post-training-compass.md)\n"
            theme_content += "- [为何所有后训练流程都从SFT开始](sft-start.md)\n"
            theme_content += "- [从SFT到偏好优化](preference-optimization.md)\n"
            theme_content += "- [走向在线策略](online-policy.md)\n"
        elif "infrastructure" in theme['id']:
            theme_content += "- [GPU内部架构](gpu-architecture.md)\n"
            theme_content += "- [GPU通信](gpu-communication.md)\n"
            theme_content += "- [构建弹性训练系统](resilient-systems.md)\n"
            theme_content += "- [优化训练吞吐量](optimizing-throughput.md)\n"

        theme_filepath = os.path.join(output_dir, f"{theme['id']}.md")
        with open(theme_filepath, 'w', encoding='utf-8') as f:
            f.write(theme_content)

        print(f"Created thematic section: {theme_filepath}")

if __name__ == "__main__":
    input_file = "/Users/peyton/Workspace/smol_training/README_new.md"

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        exit(1)

    print("Splitting markdown file into semantic sections...")
    sections = split_markdown_into_sections(input_file)

    print("\nCreating thematic summaries...")
    create_summary_sections(input_file)

    print("\nDone! 🎉")