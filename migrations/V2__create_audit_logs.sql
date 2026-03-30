-- Create audit logs table
CREATE TABLE IF NOT EXISTS audit_logs (
    id BIGSERIAL PRIMARY KEY,
    api_key_id UUID REFERENCES api_keys(id) ON DELETE SET NULL,
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER NOT NULL,
    latency_ms INTEGER NOT NULL,
    request_size_bytes BIGINT,
    response_size_bytes BIGINT,
    error_message TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_audit_logs_api_key_id ON audit_logs(api_key_id);
CREATE INDEX idx_audit_logs_created_at ON audit_logs(created_at);
CREATE INDEX idx_audit_logs_endpoint ON audit_logs(endpoint);
CREATE INDEX idx_audit_logs_status_code ON audit_logs(status_code);
