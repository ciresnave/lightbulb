-- Rate limiting usage table (fixed-window by minute)
CREATE TABLE IF NOT EXISTS api_key_usage (
    api_key_id UUID NOT NULL REFERENCES api_keys(id) ON DELETE CASCADE,
    window_start TIMESTAMP NOT NULL,
    request_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (api_key_id, window_start)
);

CREATE INDEX idx_api_key_usage_window_start ON api_key_usage(window_start);
