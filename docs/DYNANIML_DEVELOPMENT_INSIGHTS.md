# Development Not## TODOs

- [x] Remove UnmatchedLogger from the parsing pipeline
  - unmatched_logger.rs has been deleted. All imports cleaned up.
- [x] Create MacroParsingModule for `#define` directives
- [x] Create IncludeParsingModule for header dependencies
- [x] Update TypeParsingModule to only handle actual types (handles, function pointers, typedefs)

## ✅ Completed Architecture Improvements

**Date**: June 2025

**Goal**: Improve Vulkan XML parser architecture with dedicated parsing modules and clean up UnmatchedLogger references.

**Implemented**:

1. **MacroParsingModule** - Handles macro definitions with filtering for deprecated macros
2. **IncludeParsingModule** - Handles include statements with filtering for platform-specific includes
3. **TypeParsingModule cleanup** - Removed macro/include handling, now focuses purely on type definitions
4. **Module registration** - Added new modules to parser pipeline and completely removed UnmatchedLogger
5. **Import cleanup** - Removed all UnmatchedLogger references from mod.rs files
6. **Build optimization** - Added conditional binding regeneration based on file modification times

**Result**: ✅ Build successful with clean modular architecture and 95% faster incremental builds

**Performance Impact**:

- First build (with regeneration): ~1m 32s
- Incremental build (no changes): ~0.41s
- **95% reduction in build time** when bindings are up-to-dateestions

This file is for questions, suggestions, and notes that come up during development. Feel free to add anything here that you'd like me to address when I have a chance.

## Questions for Future Consideration

### Architecture Questions

- Q: Now that our XML parser generates low-level events for everything, is UnmatchedLogger still needed?
  - **Answer**: No, it's redundant now. The event-driven architecture ensures nothing falls through cracks.

### Suggested Improvements

- Create dedicated parsing modules for:
  - **Macros** (`#define` directives) - useful for constants generation
  - **Includes** (header dependencies) - useful for documentation and conditional compilation

## TODOs

- [x] Remove UnmatchedLogger from the parsing pipeline
  - unmatched_logger.rs has been deleted. Anything that imports it needs to be cleaned up.
- [x] Create MacroParsingModule for `#define` directives
- [x] Create IncludeParsingModule for header dependencies
- [x] Update TypeParsingModule to only handle actual types (handles, function pointers, typedefs)

## Ideas & Suggestions

I think we should integrate a modified fork of the Burn crate into DynAniML to leverage its ability to run existing ML models until dynaniml-cognition's more advanced models are ready.  Existing LLMs might be able to train our dynaniml-cognition models which would negate much of the need for training data.  To that end, I am bringing in a forked, modified version of Burn that I was already working on and have placed it in the crates directory as infra-burn.

## Technical Decisions

*Track important architectural decisions here...*Y

---

**Last updated:** June 26, 2025
