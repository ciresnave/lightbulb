-- Drops the dead user_sessions table (V3). Nothing in src/ ever read or wrote it
-- (PersistAnt adoption design step 6, "its own later PR"; ROADMAP A9). DROP TABLE
-- also drops its four indexes. IRREVERSIBLE: rows are lost; there were never any
-- writers. A new migration, never an edit of V3, so databases already at V3-V7
-- converge and PersistAnt's ledger records it.
DROP TABLE IF EXISTS user_sessions;
