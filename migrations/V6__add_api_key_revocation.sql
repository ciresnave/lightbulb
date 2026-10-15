-- Adds revocation support to API keys. Previously a key could be created
-- but never revoked or rotated -- the only way to disable one was to delete
-- the row directly in the database, outside any API path.
ALTER TABLE api_keys ADD COLUMN revoked_at TIMESTAMP;

-- NOT on the hot path: the per-request auth lookup in auth_middleware.rs
-- keys on key_hash (already UNIQUE, already indexed), so "AND revoked_at
-- IS NULL" there is a filter on the single row that index already found,
-- not a second index scan -- it stays one indexed read regardless of this
-- index. This index serves the cold paths instead: an admin listing
-- revoked keys, and revoke_api_key's lockout check (COUNT of live admin
-- keys), neither of which goes through key_hash at all.
CREATE INDEX idx_api_keys_revoked_at ON api_keys(revoked_at);
