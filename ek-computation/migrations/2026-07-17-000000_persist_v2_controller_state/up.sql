CREATE TABLE controller_v2_placement (
    worker_id TEXT PRIMARY KEY,
    instance_id BIGINT NOT NULL CHECK (instance_id > 0),
    generation BIGINT NOT NULL CHECK (generation > 0),
    targets JSONB NOT NULL CHECK (jsonb_typeof(targets) = 'array')
);

CREATE TABLE controller_v2_topology (
    instance_id BIGINT PRIMARY KEY CHECK (instance_id > 0),
    version BIGINT NOT NULL CHECK (version > 0)
);
