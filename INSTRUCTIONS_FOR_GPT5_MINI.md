# Instructions for GPT-5 Mini: Create Literature Summaries

## Goal
Create high-quality summary files for all PDF papers in the `docs/papers/markdown/` directory. Each summary should follow the established format and link to both the original PDF and the markdown conversion.

## Context
You're helping build "lightbulb" - a modern ML model training and inference library on top of Candle (Rust). The summaries help identify techniques, algorithms, and architectural patterns relevant to the implementation.

## Input Files
- **Markdown papers**: `docs/papers/markdown/*.md` (already converted from PDFs)
- **PDF index**: `docs/papers/pdf-index.txt` (lists all PDFs in order)
- **Existing summaries**: `docs/summaries/` (for reference on format/style)

## Output Files
- **Summaries**: `docs/summaries/<slug>.md` (one per paper)
  - Slug is the same as the markdown filename (e.g., `efficientreasoningmodels-asurvey.md`)

## Summary Format Template

```markdown
# [Paper Title]

**Full PDF:** [View Original](<C:\Users\cires\OneDrive\Desktop\books and courses\Machine Learning\[original-filename].pdf>)  
**Markdown:** [View Markdown](../papers/markdown/[slug].md)

## TL;DR
[2-3 sentences capturing the core contribution and approach]

## Why it matters
- [3-5 bullet points explaining relevance to lightbulb project]
- [Focus on: performance, efficiency, novel algorithms, architectural patterns]
- [Connect to our goals: efficient training/inference on Candle]

## Key technical takeaways
1. [Specific technique/algorithm with brief explanation]
2. [Another technique - be concrete and technical]
3. [Include numbers, metrics, or comparative results where available]
4. [Focus on implementation details that matter]
5. [Highlight trade-offs or design decisions]

## Implementation steps for lightbulb
- [Actionable step referencing specific component/module]
- [Another step with clear deliverable]
- [Include prototyping, benchmarking, or testing steps]
- [Reference telemetry, logging, or observability needs]
- [Connect to existing codebase where applicable]

## Acceptance criteria
- [Measurable outcome or performance target]
- [Another testable criterion]
- [Include latency, throughput, accuracy, or memory metrics]
- [Specify integration tests or benchmarks]
```

## Example: Efficient Reasoning Models Survey

See `docs/summaries/efficientreasoningmodels-asurvey.md` for a complete example following this format.

### Key points from this example:
- **TL;DR** is concise but captures the taxonomy (shorter/smaller/faster)
- **Why it matters** connects to computational efficiency and our Candle-based goals
- **Key technical takeaways** are numbered, specific, and implementation-focused
- **Implementation steps** are actionable and reference actual components (kernels, telemetry, etc.)
- **Acceptance criteria** include measurable metrics (<5% latency overhead, <20% accuracy degradation)

## Step-by-Step Process

### Step 1: Get the list of papers to process
```bash
cd c:\Users\cires\OneDrive\Documents\projects\lightbulb
python scripts\next_unprocessed.py
```

This shows the next 6 unprocessed PDFs with their slugs.

### Step 2: For each paper in the batch

1. **Read the markdown file**:
   ```
   File: docs/papers/markdown/[slug].md
   ```

2. **Identify the original PDF filename**:
   - Check `docs/papers/pdf-index.txt` to find the exact PDF filename
   - Or check the markdown header/metadata

3. **Create the summary file**:
   ```
   File: docs/summaries/[slug].md
   ```
   
4. **Write the summary following the template**:
   - Start with title and links (PDF + markdown)
   - Fill in each section thoughtfully
   - Focus on technical depth and implementation relevance
   - Use the example as a style guide

### Step 3: Update the indices

After creating summaries for a batch, update:

1. **README.md**: Add links to new summaries in the "Literature Review" section
   ```markdown
   - [Efficient Reasoning Models: A Survey](docs/summaries/efficientreasoningmodels-asurvey.md)
   ```

2. **docs/literature/index.md**: Update the literature index with new entries

### Step 4: Verify and iterate

- Check that all summaries follow the format
- Ensure links work (PDF and markdown files exist)
- Verify technical accuracy and depth
- Run `python scripts/next_unprocessed.py` to get the next batch
- Repeat until all 180+ papers are processed

## Quality Guidelines

### DO:
- ✅ Be specific and technical (mention algorithms, data structures, metrics)
- ✅ Focus on implementation relevance (how does this help build lightbulb?)
- ✅ Include quantitative results (speedups, accuracy, memory usage)
- ✅ Connect to Candle/Rust ecosystem when applicable
- ✅ Make implementation steps actionable (prototype X, benchmark Y)
- ✅ Write measurable acceptance criteria

### DON'T:
- ❌ Be vague or generic ("this paper is interesting")
- ❌ Just summarize the abstract
- ❌ Ignore implementation details
- ❌ Skip the "why it matters" connection to lightbulb
- ❌ Write implementation steps that are too high-level
- ❌ Create unmeasurable acceptance criteria

## Batch Processing Strategy

Process in batches of 6 papers (as shown by `next_unprocessed.py`):
1. Run the script to get next batch
2. Create 6 summaries
3. Update indices (README, literature/index.md)
4. Commit the batch
5. Repeat

This keeps work organized and allows for incremental progress tracking.

## Notes

- The markdown files are already converted (by `scripts/convert_all_pdfs.py`)
- Focus on creating high-quality summaries, not converting PDFs
- The summaries are the main deliverable for the literature review phase
- After all summaries are complete, we'll move to implementation planning

## Questions?

If you encounter:
- Missing markdown files → they may still be converting
- Unclear paper content → mark it for manual review
- Very short papers → create proportionally shorter summaries
- Non-technical content → focus on any algorithmic or systems insights

---

**Ready to start?** Run `python scripts\next_unprocessed.py` and begin with the first paper in the batch!
