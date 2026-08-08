#!/bin/bash
set -euo pipefail

# =============================================================================
# PGRL single-container pipeline for the demo TPC-H machine
#
# Flow:
#   1. Initialize/start local PostgreSQL (first run also creates TPC-H SF=10)
#   2. Sequentially train each query with PPO
#   3. Run regression-aware Global Configuration Merge (GCM)
#   4. Validate C_final on the complete workload
#
# PostgreSQL and all PGRL scripts run in this same container. No SSH tunnel and
# no per-query/parallel training containers are used.
# =============================================================================

echo "========================================="
echo "   PGRL Demo Pipeline Started"
echo "========================================="

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PG_DATA="${PGDATA:-/var/lib/postgresql/data}"
DB_NAME="${TARGET_DB_NAME:-tpch10}"
DB_USER="${TARGET_DB_USER:-wettin}"
DB_PASS="${TARGET_DB_PASS:-}"
DB_PORT="${TARGET_DB_PORT:-5432}"
TUNING_CONF="${REMOTE_CONF_PATH:-${PG_DATA}/auto_tuning.conf}"
TPCH_SCALE_FACTOR="${TPCH_SCALE_FACTOR:-10}"
TPCH_DIR="${TPCH_DIR:-/opt/tpch-dbgen/dbgen}"
PG_TPCH_DIR="${PG_TPCH_DIR:-/opt/pg_tpch}"
REPORT_DIR="${REPORT_DIR:-/app/training_log/experiment_reports}"
MERGE_DIR="${MERGE_DIR:-/app/merge_test_log}"
REFERENCE_DIR="${REFERENCE_DIR:-/app/reference_configs}"
P1_TIMESTEPS="${P1_TIMESTEPS:-2048}"
P2_TIMESTEPS="${P2_TIMESTEPS:-2048}"
P3_TIMESTEPS="${P3_TIMESTEPS:-2048}"
MAX_PARALLEL_WORKERS="${MAX_PARALLEL_WORKERS:-20}"
LOAD_MAINTENANCE_WORK_MEM="${LOAD_MAINTENANCE_WORK_MEM:-2GB}"

# Only simple PostgreSQL identifiers are accepted because the names are used in
# CREATE ROLE / CREATE DATABASE statements during first-time initialization.
if [[ ! "$DB_NAME" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
    echo "[Error] TARGET_DB_NAME contains unsupported characters: $DB_NAME" >&2
    exit 1
fi
if [[ ! "$DB_USER" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
    echo "[Error] TARGET_DB_USER contains unsupported characters: $DB_USER" >&2
    exit 1
fi
if [[ ! "$DB_PORT" =~ ^[0-9]+$ ]]; then
    echo "[Error] TARGET_DB_PORT must be numeric: $DB_PORT" >&2
    exit 1
fi
if [[ ! "$LOAD_MAINTENANCE_WORK_MEM" =~ ^[0-9]+(kB|MB|GB)$ ]]; then
    echo "[Error] LOAD_MAINTENANCE_WORK_MEM must look like 512MB or 2GB." >&2
    exit 1
fi
if [[ "$(dirname "$TUNING_CONF")" != "$PG_DATA" ]]; then
    echo "[Error] REMOTE_CONF_PATH must be inside PGDATA in local-container mode." >&2
    echo "        PGDATA=$PG_DATA" >&2
    echo "        REMOTE_CONF_PATH=$TUNING_CONF" >&2
    exit 1
fi

# Locate PostgreSQL binaries installed in the image.
PG_BINDIR="${PG_BINDIR:-}"
if [[ -z "$PG_BINDIR" ]]; then
    PG_BINDIR="$(dirname "$(find /usr/lib/postgresql -type f -name pg_ctl -print -quit)")"
fi
if [[ -z "$PG_BINDIR" || ! -x "$PG_BINDIR/pg_ctl" || ! -x "$PG_BINDIR/initdb" ]]; then
    echo "[Error] PostgreSQL pg_ctl/initdb could not be found." >&2
    exit 1
fi
PG_CTL="$PG_BINDIR/pg_ctl"
INITDB="$PG_BINDIR/initdb"

mkdir -p "$REPORT_DIR" "$MERGE_DIR" "$REFERENCE_DIR" "$PG_DATA"
chown -R postgres:postgres "$PG_DATA"
chmod 700 "$PG_DATA"

# libpq variables used by test_ppo.py, merge_configs.py, and verify_configs.py.
# All database connections stay inside this container.
export PGDATABASE="$DB_NAME"
export PGUSER="$DB_USER"
export PGPASSWORD="$DB_PASS"
export PGHOST="127.0.0.1"
export PGPORT="$DB_PORT"
export PGDATA="$PG_DATA"
export REMOTE_CONF_PATH="$TUNING_CONF"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
pg_ctl_as_postgres() {
    local action="$1"
    su - postgres -c "'$PG_CTL' -D '$PG_DATA' -w '$action'"
}

psql_super() {
    psql -h 127.0.0.1 -p "$DB_PORT" -U postgres -v ON_ERROR_STOP=1 "$@"
}

postgres_is_running() {
    su - postgres -c "'$PG_CTL' -D '$PG_DATA' status" >/dev/null 2>&1
}

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    if postgres_is_running; then
        echo "[System] Stopping PostgreSQL..."
        su - postgres -c "'$PG_CTL' -D '$PG_DATA' -m fast -w stop" >/dev/null 2>&1 || true
    fi
    exit "$exit_code"
}
trap cleanup EXIT INT TERM

# -----------------------------------------------------------------------------
# Step 0: Initialize PostgreSQL cluster and TPC-H database
# -----------------------------------------------------------------------------
echo ""
echo "[Step 0] Preparing local PostgreSQL and TPC-H database..."

if [[ ! -f "$PG_DATA/PG_VERSION" ]]; then
    echo "[Init] PostgreSQL data directory is not initialized. Running initdb..."
    # The directory must be empty for initdb.
    if [[ -n "$(find "$PG_DATA" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
        echo "[Error] $PG_DATA is non-empty but does not contain PG_VERSION." >&2
        echo "        Remove/recreate the Docker volume before retrying." >&2
        exit 1
    fi

    su - postgres -c "'$INITDB' -D '$PG_DATA'"

    cat >> "$PG_DATA/pg_hba.conf" <<'PGHBA'
local all all trust
host  all all 127.0.0.1/32 trust
PGHBA

    cat >> "$PG_DATA/postgresql.conf" <<EOF_CONF
listen_addresses = '127.0.0.1'
port = ${DB_PORT}
include = 'auto_tuning.conf'

# Fixed demo-environment settings. These are not PGRL tuning parameters.
max_wal_size = '4GB'
checkpoint_timeout = '30min'
max_worker_processes = ${MAX_PARALLEL_WORKERS}
max_parallel_workers = ${MAX_PARALLEL_WORKERS}
EOF_CONF
fi

# Ensure the tuning file and include directive also exist for an already-created
# PostgreSQL volume.
touch "$TUNING_CONF"
chown postgres:postgres "$TUNING_CONF"
chmod 644 "$TUNING_CONF"
MAIN_CONF="$PG_DATA/postgresql.conf"
TUNING_BASENAME="$(basename "$TUNING_CONF")"
if ! grep -Eq "^[[:space:]]*include[[:space:]]*=[[:space:]]*'${TUNING_BASENAME}'[[:space:]]*$" "$MAIN_CONF"; then
    echo "include = '${TUNING_BASENAME}'" >> "$MAIN_CONF"
fi

# Every pipeline run starts with an empty PGRL tuning file so that C_default is
# the underlying PostgreSQL/demo configuration rather than settings left by a
# previous interrupted run.
: > "$TUNING_CONF"
chown postgres:postgres "$TUNING_CONF"

if ! postgres_is_running; then
    echo "[System] Starting PostgreSQL..."
    pg_ctl_as_postgres start
fi

for attempt in {1..30}; do
    if pg_isready -h 127.0.0.1 -p "$DB_PORT" -d postgres >/dev/null 2>&1; then
        break
    fi
    if [[ "$attempt" -eq 30 ]]; then
        echo "[Error] PostgreSQL did not become ready." >&2
        exit 1
    fi
    sleep 1
done

# Create role if needed.
if ! psql_super -d postgres -tAc "SELECT 1 FROM pg_roles WHERE rolname = '$DB_USER'" | grep -q 1; then
    echo "[Init] Creating PostgreSQL role: $DB_USER"
    psql_super -d postgres -c "CREATE ROLE \"$DB_USER\" LOGIN SUPERUSER;"
fi

# Create database if needed.
if ! psql_super -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname = '$DB_NAME'" | grep -q 1; then
    echo "[Init] Creating database: $DB_NAME"
    psql_super -d postgres -c "CREATE DATABASE \"$DB_NAME\" OWNER \"$DB_USER\";"
fi

TPCH_MARKER="$PG_DATA/.pgrl_tpch_initialized"
if [[ -f "$TPCH_MARKER" ]]; then
    EXISTING_SF="$(cat "$TPCH_MARKER" 2>/dev/null || true)"
    if [[ "$EXISTING_SF" != "$TPCH_SCALE_FACTOR" ]]; then
        echo "[Error] Existing TPC-H database uses scale factor '$EXISTING_SF'," >&2
        echo "        but TPCH_SCALE_FACTOR='$TPCH_SCALE_FACTOR' was requested." >&2
        echo "        Remove the PostgreSQL Docker volume to rebuild the dataset." >&2
        exit 1
    fi
    echo "[Init] TPC-H SF=${TPCH_SCALE_FACTOR} already initialized. Skipping data generation."
else
    echo "[Init] Creating TPC-H schema..."
    psql_super -d "$DB_NAME" -f "$TPCH_DIR/dss.ddl"

    echo "[Init] Generating TPC-H data (SF=${TPCH_SCALE_FACTOR})..."
    rm -f "$TPCH_DIR"/*.tbl
    (
        cd "$TPCH_DIR"
        ./dbgen -vf -s "$TPCH_SCALE_FACTOR"
    )

    # Standard dbgen output has a trailing '|' on every row. Stream each file
    # through sed into client-side \copy, avoiding a second multi-GB rewritten
    # copy of the SF=10 data on disk.
    echo "[Init] Loading TPC-H tables..."
    for table in customer lineitem nation orders part partsupp region supplier; do
        file_path="$TPCH_DIR/$table.tbl"
        if [[ ! -f "$file_path" ]]; then
            echo "[Error] Missing generated file: $file_path" >&2
            exit 1
        fi
        echo "       -> $table"
        sed 's/|$//' "$file_path" | \
            psql -h 127.0.0.1 -p "$DB_PORT" -U postgres -d "$DB_NAME" -v ON_ERROR_STOP=1 \
                -c "\\copy $table FROM STDIN WITH (FORMAT csv, DELIMITER '|');"
        rm -f "$file_path"
    done

    echo "[Init] Creating primary keys..."
    PGOPTIONS="-c maintenance_work_mem=${LOAD_MAINTENANCE_WORK_MEM}" \
        psql -h 127.0.0.1 -p "$DB_PORT" -U postgres -d "$DB_NAME" -v ON_ERROR_STOP=1 \
            -f "$PG_TPCH_DIR/dss/tpch-pkeys.sql"

    echo "[Init] Creating indexes..."
    PGOPTIONS="-c maintenance_work_mem=${LOAD_MAINTENANCE_WORK_MEM}" \
        psql -h 127.0.0.1 -p "$DB_PORT" -U postgres -d "$DB_NAME" -v ON_ERROR_STOP=1 \
            -f "$PG_TPCH_DIR/dss/tpch-index.sql"

    echo "[Init] Running VACUUM ANALYZE..."
    psql_super -d "$DB_NAME" -c "VACUUM ANALYZE;"

    # Generated source files are no longer needed after COPY.
    rm -f "$TPCH_DIR"/*.tbl
    echo "$TPCH_SCALE_FACTOR" > "$TPCH_MARKER"
    chown postgres:postgres "$TPCH_MARKER"
    echo "[Init] TPC-H initialization completed."
fi

# -----------------------------------------------------------------------------
# Workload selection
# -----------------------------------------------------------------------------
if [[ -n "${PGRL_QUERIES:-}" ]]; then
    IFS=',' read -r -a QUERIES <<< "$PGRL_QUERIES"
else
    QUERIES=(
        Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9 Q10 Q11
        Q12 Q13 Q14 Q15 Q16 Q17 Q18 Q19 Q20 Q21 Q22
    )
fi

# Trim accidental spaces and reject empty query names.
NORMALIZED_QUERIES=()
for q in "${QUERIES[@]}"; do
    q="${q//[[:space:]]/}"
    [[ -n "$q" ]] && NORMALIZED_QUERIES+=("$q")
done
QUERIES=("${NORMALIZED_QUERIES[@]}")
if [[ "${#QUERIES[@]}" -eq 0 ]]; then
    echo "[Error] No workload queries were configured." >&2
    exit 1
fi
QUERIES_STR="$(IFS=,; echo "${QUERIES[*]}")"

# Validate query names before beginning a potentially long training run.
PGRL_QUERY_LIST="$QUERIES_STR" python - <<'PY'
import os
from src.tpch_queryspecs import SQL_MAP
names = [q for q in os.environ["PGRL_QUERY_LIST"].split(",") if q]
missing = [q for q in names if q not in SQL_MAP]
if missing:
    raise SystemExit(f"Unknown query name(s) in PGRL_QUERIES: {', '.join(missing)}")
print(f"[Config] Validated {len(names)} TPC-H queries.")
PY

# The empty reference configuration represents the underlying PostgreSQL/demo
# baseline when verify_configs.py applies it.
DEFAULT_REF="$REFERENCE_DIR/postgresql_default.conf"
if [[ ! -f "$DEFAULT_REF" ]]; then
    cat > "$DEFAULT_REF" <<'EOF_REF'
# Empty by design.
# Applying this file clears auto_tuning.conf and evaluates the underlying
# PostgreSQL/demo configuration as the validation baseline.
EOF_REF
fi

# -----------------------------------------------------------------------------
# Step 1: Sequential query-level PPO training
# -----------------------------------------------------------------------------
echo ""
echo "[Config] PostgreSQL: ${DB_NAME} (user=${DB_USER}, port=${DB_PORT})"
echo "[Config] PGDATA: ${PG_DATA}"
echo "[Config] Tuning file: ${TUNING_CONF}"
echo "[Config] TPC-H scale factor: ${TPCH_SCALE_FACTOR}"
echo "[Config] Workload size: ${#QUERIES[@]} queries"
echo "[Config] Timesteps: P1=${P1_TIMESTEPS}, P2=${P2_TIMESTEPS}, P3=${P3_TIMESTEPS}"

echo ""
echo "[Step 1] Starting sequential query-level training..."
for q in "${QUERIES[@]}"; do
    echo ""
    echo ">>> Training ${q}"
    python src/test_ppo.py \
        --dsn "application_name=PGRL-TRAIN-DEMO" \
        --remote-db-port "$DB_PORT" \
        --remote-conf "$TUNING_CONF" \
        --queries "$q" \
        --total-p1 "$P1_TIMESTEPS" \
        --total-p2 "$P2_TIMESTEPS" \
        --total-p3 "$P3_TIMESTEPS" \
        --report-dir "$REPORT_DIR"
done

# -----------------------------------------------------------------------------
# Step 2: Regression-aware GCM
# -----------------------------------------------------------------------------
echo ""
echo "[Step 2] Executing regression-aware Global Configuration Merge..."
echo "[Step 2] lambda=${DEGRADATION_LAMBDA:-2.5}, power=${DEGRADATION_POWER:-2.0}"
python src/merge_configs.py \
    --dsn "application_name=PGRL-MERGE-DEMO" \
    --remote-db-port "$DB_PORT" \
    --remote-conf "$TUNING_CONF" \
    --report-dir "$REPORT_DIR" \
    --degradation-lambda "${DEGRADATION_LAMBDA:-2.5}" \
    --degradation-power "${DEGRADATION_POWER:-2.0}" \
    --queries "$QUERIES_STR"

# -----------------------------------------------------------------------------
# Step 3: Final validation
# -----------------------------------------------------------------------------
echo ""
echo "[Step 3] Starting final validation..."
python src/verify_configs.py \
    --dsn "application_name=PGRL-VERIFY-DEMO" \
    --remote-db-port "$DB_PORT" \
    --remote-conf "$TUNING_CONF" \
    --ref-dir "$REFERENCE_DIR" \
    --merged-dir "$MERGE_DIR" \
    --queries "$QUERIES_STR"

echo ""
echo "========================================="
echo "   Full PGRL demo pipeline completed"
echo "========================================="
