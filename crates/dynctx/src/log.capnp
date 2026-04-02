@0xb17db581dbaa82aa;
struct LogEntry {
  seq      @0 :UInt64;
  op       @1 :UInt16;  # see Op enum in Rust
  payload  @2 :Data;    # compressed or raw bytes
}
