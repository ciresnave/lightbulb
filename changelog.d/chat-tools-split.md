kind: maintenance: tool-call handling moved to chat/tools.rs, portfolio pin 0.48.1, ROADMAP section H
---
G4: 538 lines of tool-call handling moved verbatim from `api/openai/chat.rs` into `api/openai/chat/tools.rs` (no behaviour change; the 132 `api::openai` lib tests pass). CI pin moved to coderipper#375 (installs CodeRipper 0.48.1). ROADMAP gains section H, one row per open docs-drift item.
