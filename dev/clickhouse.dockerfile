FROM clickhouse:25.4

RUN cat > /docker-entrypoint-initdb.d/init-db.sh  <<EOF
#!/bin/bash
set -e

clickhouse client -n <<-EOSQL
    CREATE DATABASE IF NOT EXISTS dev;
    CREATE TABLE dev.expert_activate (
        worker      String,
        model       String,
        layer       UInt8,
        idx         UInt8,
        count       UInt64
    ) ENGINE = ReplacingMergeTree
    PRIMARY KEY (worker,model,layer,idx);

    CREATE TABLE dev.expert_activate_detail (
        worker      String,
        model       String,
        layer       UInt8,
        idx         UInt8,
        activate_at DateTime64(3)  DEFAULT now(),
        count       UInt64
    ) ENGINE = ReplacingMergeTree
    PRIMARY KEY (worker,model,layer,idx,activate_at);
EOSQL

EOF