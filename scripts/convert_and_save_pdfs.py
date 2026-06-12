#!/usr/bin/env python3
"""
Converts PDFs to markdown using the MCP tool and saves them.
This helps process multiple PDFs efficiently.
"""
import os
import sys

# List of PDFs to convert (from next_unprocessed.py output)
pdfs_to_convert = [
    {
        "path": r"c:\Users\cires\OneDrive\Desktop\books and courses\Machine Learning\Encyclopedia_Machine_Learning_2011.pdf",
        "slug": "encyclopedia-machine-learning-2011.md",
    },
    {
        "path": r"c:\Users\cires\OneDrive\Desktop\books and courses\Machine Learning\Evaluatory XAI.pdf",
        "slug": "evaluatory-xai.md",
    },
    {
        "path": r"c:\Users\cires\OneDrive\Desktop\books and courses\Machine Learning\ExploringTheLimitOfOutcomeRewardForLearningMathematicalReasoning.pdf",
        "slug": "exploringthelimitofoutcomerewardforlearningmathematicalreasoning.md",
    },
    {
        "path": r"c:\Users\cires\OneDrive\Desktop\books and courses\Machine Learning\ExplosiveNeuralNetworksViaHigherOrderInteractionsInCurvedStatisticalManifolds.pdf",
        "slug": "explosiveneuralnetworksviahigherorderinteractionsincurvedstatisticalmanifolds.md",
    },
]

output_dir = (
    r"c:\Users\cires\OneDrive\Documents\projects\lightbulb\docs\papers\markdown"
)


def main():
    print(f"Will convert {len(pdfs_to_convert)} PDFs to markdown...")
    print(f"Output directory: {output_dir}")

    for i, pdf_info in enumerate(pdfs_to_convert, 1):
        pdf_path = pdf_info["path"]
        slug = pdf_info["slug"]
        output_path = os.path.join(output_dir, slug)

        print(
            f"\n[{i}/{len(pdfs_to_convert)}] Processing: {os.path.basename(pdf_path)}"
        )
        print(f"  -> {slug}")

        # Check if already exists
        if os.path.exists(output_path):
            print(f"  ✓ Already exists, skipping")
            continue

        print(f"  Need to convert this PDF using MCP tool")
        print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
