-- Pre-auth, connection-keyed attempt counter (fixed-window by minute).
--
-- Keyed by client IP, not by api_key_id: this limiter runs BEFORE the
-- bearer-token hash+lookup (auth_middleware), so there is no key identity
-- yet to key on -- that is the whole point, since a brute-force guessing
-- campaign never presents a valid key. Rows age out via a periodic cleanup
-- job (ApiServer::new spawns one when a database is configured) rather than
-- growing unbounded.
CREATE TABLE IF NOT EXISTS auth_attempt_counts (
    client_ip INET NOT NULL,
    window_start TIMESTAMP NOT NULL,
    request_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (client_ip, window_start)
);

CREATE INDEX idx_auth_attempt_counts_window_start ON auth_attempt_counts(window_start);
