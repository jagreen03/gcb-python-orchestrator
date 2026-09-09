-- =============================================================================
-- Cyclopter persistence layer - migration 001
-- =============================================================================
-- Instantiates gcb-wiki/09_pbg_telemetry_vector_space.md section 5.1 (table) and
-- section 5.2 (indexes) VERBATIM. This migration does not redefine the schema
-- (column set, vector dimension, distance ops) - it creates it.
--
-- Idempotent: safe to run on every `load_telemetry_pg.py` invocation.
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- ---- 5.1 Table -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pbg_telemetry (
    ticket_id       text PRIMARY KEY,           -- 'TKT-003'
    title           text        NOT NULL,
    status          text        NOT NULL,       -- 'open' | 'in-progress' | 'done'
    telemetry_vec   vector(3)   NOT NULL,       -- (spec X, quality Y, emergence Z), each in [-1, 1]
    telemetry_color text        NOT NULL,       -- derived dominant spoke, or 'pending'
    vec_norm        real GENERATED ALWAYS AS (   -- ||T||_2, in [0, sqrt(3)]
                        sqrt( telemetry_vec[1]*telemetry_vec[1]
                            + telemetry_vec[2]*telemetry_vec[2]
                            + telemetry_vec[3]*telemetry_vec[3] )
                    ) STORED,
    updated_at      timestamptz NOT NULL DEFAULT now()
);

-- ---- 5.2 Indexes and distance metric ------------------------------------
-- Primary - cosine: clusters by direction / flavour. Cosine is undefined at
-- the origin, so 'pending' rows are excluded from this index.
CREATE INDEX IF NOT EXISTS pbg_telemetry_cos ON pbg_telemetry
    USING hnsw (telemetry_vec vector_cosine_ops)
    WHERE telemetry_color <> 'pending';

-- Secondary - L2: flavour AND intensity, and includes the origin.
CREATE INDEX IF NOT EXISTS pbg_telemetry_l2 ON pbg_telemetry
    USING hnsw (telemetry_vec vector_l2_ops);
