# Changelog

## 0.4.2

### Changed
- The KV-cache disk tier (`cache::tiered_storage::FileDiskStore`) now writes atomically through
  `persistant` 0.4.1 (crates.io) instead of raw `std::fs` calls, and creates a sibling
  `<dir>.atomic-scratch` directory beside every `FileDiskStore` base directory — a new on-disk
  artifact, swept of any abandoned temp file at open. Reading files left behind by the old writer
  is unaffected: `FileDiskStore` still reads whatever plain file is at `base_dir/<key>`.
  - `base_dir` is resolved (symlinks and relative components, including a bare `.`) before the
    scratch sibling is derived from it, so the sibling stays outside the real root directory.
  - The sweep only runs once this `FileDiskStore` holds the scratch directory's lock file
    exclusively (a fresh `create_new`) — a scratch directory another live store still holds is
    left untouched rather than swept blind.
