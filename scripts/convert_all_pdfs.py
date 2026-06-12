#!/usr/bin/env python3
"""
Convert all PDFs to markdown files using markitdown CLI.

This script:
1. Reads pdf-index.txt to get all PDFs in order
2. Checks which ones don't have markdown files yet
3. Converts missing ones using markitdown command-line tool

Install markitdown if needed:
    pip install markitdown
"""

import os
import subprocess
import sys
from pathlib import Path


def get_pdf_slug(pdf_filename):
    """Convert PDF filename to normalized slug for markdown filename."""
    # Remove extension
    name = Path(pdf_filename).stem

    # Convert to lowercase
    slug = name.lower()

    # Replace non-alphanumeric with hyphens
    import re

    slug = re.sub(r"[^a-z0-9]+", "-", slug)

    # Collapse multiple hyphens
    slug = re.sub(r"-+", "-", slug)

    # Remove leading/trailing hyphens
    slug = slug.strip("-")

    return slug + ".md"


def main():
    # Paths
    project_root = Path(__file__).parent.parent
    pdf_index_file = project_root / "docs" / "papers" / "pdf-index.txt"
    pdf_dir = Path(
        r"C:\Users\cires\OneDrive\Desktop\books and courses\Machine Learning"
    )
    markdown_dir = project_root / "docs" / "papers" / "markdown"

    # Ensure markdown directory exists
    markdown_dir.mkdir(parents=True, exist_ok=True)

    # Read PDF index
    if not pdf_index_file.exists():
        print(f"ERROR: PDF index not found at {pdf_index_file}")
        return 1

    # Try different encodings
    for encoding in ["utf-8", "utf-16", "utf-8-sig", "latin-1"]:
        try:
            with open(pdf_index_file, "r", encoding=encoding) as f:
                pdf_paths = [line.strip() for line in f if line.strip()]
            print(f"Read index with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
    else:
        print("ERROR: Could not decode pdf-index.txt")
        return 1

    print(f"Found {len(pdf_paths)} PDFs in index")

    # Check which ones need conversion
    to_convert = []
    for pdf_path_str in pdf_paths:
        pdf_path = Path(pdf_path_str)
        pdf_filename = pdf_path.name
        slug = get_pdf_slug(pdf_filename)
        markdown_path = markdown_dir / slug

        if not markdown_path.exists():
            if pdf_path.exists():
                to_convert.append((pdf_path, markdown_path))
            else:
                print(f"WARNING: PDF not found: {pdf_path}")

    print(f"\nNeed to convert {len(to_convert)} PDFs to markdown")

    if not to_convert:
        print("All PDFs already converted!")
        return 0

    # Check if markitdown is available
    try:
        subprocess.run(["markitdown", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\nERROR: markitdown not found!")
        print("Install it with: pip install markitdown")
        return 1

    # Convert PDFs
    print("\nStarting conversions...")
    success_count = 0
    error_count = 0

    for i, (pdf_path, markdown_path) in enumerate(to_convert, 1):
        print(f"\n[{i}/{len(to_convert)}] Converting: {pdf_path.name}")
        print(f"  -> {markdown_path.name}")

        try:
            # Run markitdown to convert PDF
            # Don't use text=True or encoding - capture raw bytes and decode manually
            result = subprocess.run(
                ["markitdown", str(pdf_path)],
                capture_output=True,
                check=True,
            )

            # Decode output, handling encoding errors gracefully
            try:
                output = result.stdout.decode("utf-8")
            except UnicodeDecodeError:
                # Try with error handling
                output = result.stdout.decode("utf-8", errors="replace")
                print(f"  [!] Warning: Had to replace some invalid UTF-8 characters")

            if not output or len(output) < 100:
                print(f"  [!] Warning: Output too short ({len(output)} bytes)")
                error_count += 1
                continue

            with open(markdown_path, "w", encoding="utf-8") as f:
                f.write(output)

            print(f"  [OK] Success ({len(output)} bytes)")
            success_count += 1

        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] Process failed: {e}")
            error_count += 1
        except Exception as e:
            print(f"  [ERROR] Unexpected: {e}")
            error_count += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"  Success: {success_count}")
    print(f"  Errors:  {error_count}")
    print(f"  Total:   {len(to_convert)}")
    print(f"{'='*60}")

    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
