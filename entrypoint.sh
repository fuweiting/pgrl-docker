#!/bin/bash
set -e

# Define variables
PG_DATA="/var/lib/postgresql/data"
DB_NAME="tpch10"
DB_USER="wettin"
TPCH_DIR="/opt/tpch-dbgen/dbgen"
PG_TPCH_DIR="/opt/pg_tpch"

# Ensure data directory exists
if [ ! -d "$PG_DATA" ]; then
    echo "[Init] Creating data directory: $PG_DATA"
    mkdir -p "$PG_DATA"
    chown -R postgres:postgres "$PG_DATA"
    chmod 700 "$PG_DATA"
fi

# Check if the database has already been initialized
if [ -z "$(ls -A "$PG_DATA")" ]; then
    echo "[Init] Data directory is empty. Starting initialization..."

    # 1. Initialize PostgreSQL data directory
    echo "[Init] Running initdb..."
    chown -R postgres:postgres "$PG_DATA"
    su - postgres -c "/usr/lib/postgresql/15/bin/initdb -D $PG_DATA"

    # 2. Modify configuration files
    echo "host all all 127.0.0.1/32 trust" >> "$PG_DATA/pg_hba.conf"
    echo "local all all trust" >> "$PG_DATA/pg_hba.conf"
    echo "listen_addresses = '*'" >> "$PG_DATA/postgresql.conf"
    
    # Create an empty auto_tuning.conf and include it
    touch "$PG_DATA/auto_tuning.conf"
    chown postgres:postgres "$PG_DATA/auto_tuning.conf"
    echo "include = 'auto_tuning.conf'" >> "$PG_DATA/postgresql.conf"

    # Optimize write performance (speed up index creation)
    echo "max_wal_size = 4GB" >> "$PG_DATA/postgresql.conf"
    echo "checkpoint_timeout = 30min" >> "$PG_DATA/postgresql.conf"
    echo "maintenance_work_mem = 2GB" >> "$PG_DATA/postgresql.conf"

    # 3. Temporarily start the database
    echo "[Init] Starting temp DB..."
    su - postgres -c "/usr/lib/postgresql/15/bin/pg_ctl -D $PG_DATA -w start"

    # 4. Create User and Database
    echo "[Init] Creating User $DB_USER and Database $DB_NAME..."
    su - postgres -c "psql -c \"CREATE USER $DB_USER WITH SUPERUSER;\""
    su - postgres -c "psql -c \"CREATE DATABASE $DB_NAME OWNER $DB_USER;\""

    # 5. Create Table Schema
    echo "[Init] Creating Tables..."
    su - postgres -c "psql -d $DB_NAME -f $TPCH_DIR/dss.ddl"

    # 6. Generate Data
    echo "[Init] Generating TPC-H Data (Scale Factor 10)..."
    cd $TPCH_DIR
    ./dbgen -vf -s 10
    chmod 644 $TPCH_DIR/*.tbl

    # 7. Import Data
    echo "[Init] Loading .tbl files into Database..."
    for table in customer lineitem nation orders part partsupp region supplier; do
        file_path="$TPCH_DIR/$table.tbl"
        if [ -f "$file_path" ]; then
            echo "  -> Loading $table..."
            su - postgres -c "psql -d $DB_NAME -c \"COPY $table FROM '$file_path' WITH (FORMAT csv, DELIMITER '|');\""
            rm "$file_path"
        fi
    done

    # 8. Create Primary Keys and Indexes (using tvondra/pg_tpch)
    echo "[Init] Creating Primary Keys (This takes time)..."
    su - postgres -c "psql -d $DB_NAME -f $PG_TPCH_DIR/dss/tpch-pkeys.sql"

    echo "[Init] Creating Indexes (This takes MORE time)..."
    su - postgres -c "psql -d $DB_NAME -f $PG_TPCH_DIR/dss/tpch-index.sql"
    # ----------------------------------------------------------------

    # 9. Grant permissions
    echo "[Init] Granting permissions..."
    su - postgres -c "psql -d $DB_NAME -c \"GRANT SELECT ON ALL TABLES IN SCHEMA public TO $DB_USER;\""

    # 10. Optimize database
    echo "[Init] Running VACUUM ANALYZE..."
    su - postgres -c "psql -d $DB_NAME -c \"VACUUM ANALYZE;\""

    # 11. Stop temporary DB
    echo "[Init] Initialization complete. Stopping temp DB..."
    su - postgres -c "/usr/lib/postgresql/15/bin/pg_ctl -D $PG_DATA -m fast stop"
else
    echo "[Init] Database already initialized. Skipping setup."
    if [ ! -f "$PG_DATA/auto_tuning.conf" ]; then
        touch "$PG_DATA/auto_tuning.conf"
        chown postgres:postgres "$PG_DATA/auto_tuning.conf"
    fi
fi

# Official startup phase
echo "[System] Starting PostgreSQL Server..."
mkdir -p /var/log/postgresql
chown -R postgres:postgres /var/log/postgresql

# Start DB in the background
su - postgres -c "/usr/lib/postgresql/15/bin/pg_ctl -D $PG_DATA -l /var/log/postgresql/server.log start"

# Wait for DB Ready
echo "[System] Waiting for DB to be ready..."
sleep 3

echo "[System] Database is ready. Starting Application..."
cd /app
exec "$@"