-- Create API keys table
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key_hash VARCHAR(64) NOT NULL UNIQUE,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'admin', 'llm')),
    description TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP,
    last_used_at TIMESTAMP
);

CREATE INDEX idx_api_keys_key_hash ON api_keys(key_hash);
CREATE INDEX idx_api_keys_created_at ON api_keys(created_at);
CREATE INDEX idx_api_keys_expires_at ON api_keys(expires_at);
