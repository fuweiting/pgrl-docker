#!/bin/bash
set -e

# 定義變數
PG_DATA="/var/lib/postgresql/data"
DB_NAME="tpch10"
DB_USER="wettin"
TPCH_DIR="/opt/tpch-dbgen/dbgen"

# 確保資料目錄存在，如果不存在就建立並設定權限
if [ ! -d "$PG_DATA" ]; then
    echo "[Init] Creating data directory: $PG_DATA"
    mkdir -p "$PG_DATA"
    chown -R postgres:postgres "$PG_DATA"
    chmod 700 "$PG_DATA"
fi

# 檢查資料庫是否已經初始化過
if [ -z "$(ls -A "$PG_DATA")" ]; then
    echo "[Init] Data directory is empty. Starting initialization..."

    # 1. 初始化 PostgreSQL 資料目錄
    echo "[Init] Running initdb..."
    chown -R postgres:postgres "$PG_DATA"
    su - postgres -c "/usr/lib/postgresql/15/bin/initdb -D $PG_DATA"

    # 2. 修改設定檔
    echo "host all all 127.0.0.1/32 trust" >> "$PG_DATA/pg_hba.conf"
    echo "local all all trust" >> "$PG_DATA/pg_hba.conf"
    echo "listen_addresses = '*'" >> "$PG_DATA/postgresql.conf"
    
    # [關鍵修正] 先建立空的 auto_tuning.conf，否則 DB 會因為找不到檔案而無法啟動
    touch "$PG_DATA/auto_tuning.conf"
    chown postgres:postgres "$PG_DATA/auto_tuning.conf"
    
    # 然後再告訴 Postgres 去 include 它
    echo "include = 'auto_tuning.conf'" >> "$PG_DATA/postgresql.conf"

    # 3. 暫時啟動資料庫
    echo "[Init] Starting temp DB..."
    su - postgres -c "/usr/lib/postgresql/15/bin/pg_ctl -D $PG_DATA -w start"

    # 4. 建立使用者與資料庫
    echo "[Init] Creating User $DB_USER and Database $DB_NAME..."
    su - postgres -c "psql -c \"CREATE USER $DB_USER WITH SUPERUSER;\""
    su - postgres -c "psql -c \"CREATE DATABASE $DB_NAME OWNER $DB_USER;\""

    # 5. 建立 Table Schema
    echo "[Init] Creating Tables..."
    # 假設 dss.ddl 在正確位置
    su - postgres -c "psql -d $DB_NAME -f $TPCH_DIR/dss.ddl"

    # 6. 生成資料
    echo "[Init] Generating TPC-H Data (Scale Factor 10)..."
    cd $TPCH_DIR
    ./dbgen -vf -s 10
    
    # 修正權限確保 Postgres 能讀
    chmod 644 $TPCH_DIR/*.tbl

    # 7. 匯入資料
    echo "[Init] Loading .tbl files into Database..."
    for table in customer lineitem nation orders part partsupp region supplier; do
        file_path="$TPCH_DIR/$table.tbl"
        if [ -f "$file_path" ]; then
            echo "  -> Loading $table..."
            su - postgres -c "psql -d $DB_NAME -c \"COPY $table FROM '$file_path' WITH (FORMAT csv, DELIMITER '|');\""
            rm "$file_path"
        fi
    done

    # 9. 權限設定
    echo "[Init] Granting permissions..."
    su - postgres -c "psql -d $DB_NAME -c \"GRANT SELECT ON ALL TABLES IN SCHEMA public TO $DB_USER;\""

    # 10. 優化資料庫
    echo "[Init] Running VACUUM ANALYZE..."
    su - postgres -c "psql -d $DB_NAME -c \"VACUUM ANALYZE;\""

    # 11. 停止暫時的 DB
    echo "[Init] Initialization complete. Stopping temp DB..."
    su - postgres -c "/usr/lib/postgresql/15/bin/pg_ctl -D $PG_DATA -m fast stop"
else
    echo "[Init] Database already initialized. Skipping setup."
    
    # [保險措施] 如果資料夾存在但 auto_tuning.conf 不見了（例如被手動刪除），補回來
    if [ ! -f "$PG_DATA/auto_tuning.conf" ]; then
        touch "$PG_DATA/auto_tuning.conf"
        chown postgres:postgres "$PG_DATA/auto_tuning.conf"
    fi
fi

# --- 正式啟動階段 ---

echo "[System] Starting PostgreSQL Server..."
mkdir -p /var/log/postgresql
chown -R postgres:postgres /var/log/postgresql

# 背景啟動 DB
su - postgres -c "/usr/lib/postgresql/15/bin/pg_ctl -D $PG_DATA -l /var/log/postgresql/server.log start"

# 等待 DB Ready
echo "[System] Waiting for DB to be ready..."
sleep 3

echo "[System] Database is ready. Starting Application..."
cd /app
exec "$@"