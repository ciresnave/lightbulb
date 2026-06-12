from pathlib import Path
import re
import argparse
from urllib.parse import quote

pdf_index = Path(
    r"c:\Users\cires\OneDrive\Documents\projects\lightbulb\docs\papers\pdf-index.txt"
)
summaries_dir = Path(
    r"c:\Users\cires\OneDrive\Documents\projects\lightbulb\docs\summaries"
)


def slugify(name: str) -> str:
    # remove extension if present
    name = re.sub(r"\.[Pp][Dd][Ff]$", "", name)
    # lower
    s = name.lower()
    # replace non-alnum with hyphen
    s = re.sub(r"[^a-z0-9]+", "-", s)
    # collapse
    s = re.sub(r"-+", "-", s).strip("-")
    return s + ".md"


# read file with fallback encodings to handle non-utf8 files
encodings = ["utf-8-sig", "utf-8", "utf-16", "cp1252", "latin1"]
text = None
for enc in encodings:
    try:
        text = pdf_index.read_text(encoding=enc)
        break
    except Exception:
        continue
if text is None:
    # last resort: read raw bytes and decode with replacement
    text = pdf_index.read_bytes().decode("utf-8", errors="replace")
pdfs = [line.strip() for line in text.splitlines() if line.strip()]
existing = {p.name.lower() for p in summaries_dir.iterdir() if p.is_file()}

unprocessed = []
for p in pdfs:
    basename = Path(p).stem
    candidate = slugify(basename)
    # also accept underscore variants for legacy filenames (a_b_c.md)
    candidate_alt = candidate.replace("-", "_")
    if candidate.lower() not in existing and candidate_alt.lower() not in existing:
        unprocessed.append((p, candidate))
    if len(unprocessed) >= 6:
        break

print("# Next unprocessed PDFs (up to 6):")
for pdfpath, md in unprocessed:
    print(pdfpath)
    print("->", md)

if not unprocessed:
    print("All PDFs up to the scanned index have summaries.")


def scaffold_summaries(unprocessed_list):
    """Create skeleton summary files for each (pdfpath, slug) in unprocessed_list.
    Does not overwrite existing files.
    """
    header_template = (
        "# {title}\n\n"
        "Full PDF: [Local PDF]({file_uri})\n\n"
        "Markdown: ../papers/markdown/{md_name}\n\n"
        "## TL;DR\n\n"
        "(one-line summary here)\n\n"
        "## Why it matters\n\n"
        "- (short bullet)\n\n"
        "## Key technical takeaways\n\n"
        "1. (takeaway one)\n\n"
        "## Implementation steps for Lightbulb\n\n"
        "- (concrete step)\n\n"
        "## Acceptance criteria\n\n"
        "- (measurable condition)\n"
    )

    for pdfpath, slug in unprocessed_list:
        target = summaries_dir / slug
        if target.exists():
            print(f"Skipping existing summary: {target.name}")
            continue
        title = Path(pdfpath).stem.replace("_", " ")
        # build file:// URI and quote path components
        file_uri = "file:///" + quote(str(Path(pdfpath).as_posix()))
        md_name = slug
        content = header_template.format(
            title=title, file_uri=file_uri, md_name=md_name
        )
        target.write_text(content, encoding="utf-8")
        print(f"Scaffolded: {target}")


def main():
    parser = argparse.ArgumentParser(
        description="List next unprocessed PDFs and optionally scaffold summaries."
    )
    parser.add_argument(
        "--scaffold",
        action="store_true",
        help="Create skeleton summary files for unprocessed PDFs",
    )
    args = parser.parse_args()

    if args.scaffold:
        if not unprocessed:
            print("Nothing to scaffold; no unprocessed PDFs found.")
            return
        scaffold_summaries(unprocessed)


if __name__ == "__main__":
    main()
