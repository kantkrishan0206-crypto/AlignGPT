CREATE TABLE IF NOT EXISTS experiment_runs (
    run_id TEXT PRIMARY KEY,
    config_hash TEXT NOT NULL,
    dataset_fingerprint TEXT,
    model_revision TEXT,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS safety_findings (
    finding_id TEXT PRIMARY KEY,
    run_id TEXT,
    category TEXT NOT NULL,
    severity TEXT NOT NULL,
    rule_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES experiment_runs(run_id)
);
