-- Post-auth, failures-only brute-force counter (fixed-window by minute).
--
-- Issue #130: a SEPARATE table from auth_attempt_counts (V5), which counts
-- EVERY request per IP, successful or not, and so must stay a generous
-- flood limiter rather than a brute-force detector. This one is written
-- only when auth_middleware returns a real authentication failure
-- (missing/malformed header, or a hash matching no live key) -- never on
-- a successful request, and never on a database error -- which is what
-- makes a strict threshold here safe for legitimate traffic. Rows age out
-- via a periodic cleanup job (ApiServer::new spawns one when a database is
-- configured), same as auth_attempt_counts.
CREATE TABLE IF NOT EXISTS auth_failure_counts (
    client_ip INET NOT NULL,
    window_start TIMESTAMP NOT NULL,
    failure_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (client_ip, window_start)
);

CREATE INDEX idx_auth_failure_counts_window_start ON auth_failure_counts(window_start);
