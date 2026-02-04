#!/bin/bash
set -e

# 定義變數
PG_DATA="/var/lib/postgresql/data"
DB_NAME="tpch10"
DB_USER="wettin"  # 你的同名帳號
TPCH_DIR="/opt/tpch-dbgen/dbgen"

# 檢查資料庫是否已經初始化過
if [ -z "$(ls -A "$PG_DATA")" ]; then
    echo "[Init] Data directory is empty. Starting initialization..."

    # 1. 初始化 PostgreSQL 資料目錄
    echo "[Init] Running initdb..."
    chown -R postgres:postgres "$PG_DATA"
    su - postgres -c "/usr/lib/postgresql/15/bin/initdb -D $PG_DATA"

    # 2. 修改設定檔以允許免密碼連線 (方便 Agent 操作)
    echo "host all all 127.0.0.1/32 trust" >> "$PG_DATA/pg_hba.conf"
    echo "local all all trust" >> "$PG_DATA/pg_hba.conf"
    echo "listen_addresses = '*'" >> "$PG_DATA/postgresql.conf"
    echo "include = 'auto_tuning.conf'" >> "$PG_DATA/postgresql.conf"

    # 3. 暫時啟動資料庫 (為了建立使用者和匯入資料)
    echo "[Init] Starting temp DB..."
    su - postgres -c "/usr/lib/postgresql/15/bin/pg_ctl -D $PG_DATA -w start"

    # 4. 建立使用者與資料庫
    # 對應你的: sudo -u postgres createdb tpch10
    echo "[Init] Creating User $DB_USER and Database $DB_NAME..."
    su - postgres -c "psql -c \"CREATE USER $DB_USER WITH SUPERUSER;\""
    su - postgres -c "psql -c \"CREATE DATABASE $DB_NAME OWNER $DB_USER;\""

    # 5. 建立 Table Schema (DDL)
    # 假設 dss.ddl 已經被 COPY 到 /app/dss.ddl 或 /opt/tpch-dbgen/dbgen/dss.ddl
    # 這裡我們使用 tpch-kit 內建的 dss.ddl (注意：Postgres 語法可能需要微調，如果不相容，請確保你的專案目錄有提供修正版的 dss.ddl)
    echo "[Init] Creating Tables..."
    # 這裡假設你的 dss.ddl 在 /opt/tpch-dbgen/dbgen/dss.ddl (這是官方 repo 的位置)
    # 如果你有自己修正過的 dss.ddl，請將下行路徑改成 /app/dss.ddl
    su - postgres -c "psql -d $DB_NAME -f $TPCH_DIR/dss.ddl"

    # 6. 生成資料 (Scale Factor 10)
    # 對應你的: ./dbgen -vf -s 10
    echo "[Init] Generating TPC-H Data (Scale Factor 10)... This will take a while!"
    cd $TPCH_DIR
    # 注意: dbgen 產生檔案權限問題，我們用 postgres 身份執行或確保權限
    # 這裡直接執行，稍後用 COPY 讀取
    ./dbgen -vf -s 10

    # [修正] 確保 postgres 使用者有權限讀取產生的檔案
    chmod 644 $TPCH_DIR/*.tbl

    # 7. 匯入資料 (Bulk Load)
    echo "[Init] Loading .tbl files into Database..."
    # TPC-H 的 tbl 檔案通常以 '|' 分隔，且最後一行也有分隔符，Postgres COPY 能夠處理
    for table in customer lineitem nation orders part partsupp region supplier; do
        file_path="$TPCH_DIR/$table.tbl"
        if [ -f "$file_path" ]; then
            echo "  -> Loading $table..."
            # 使用 COPY 指令匯入
            su - postgres -c "psql -d $DB_NAME -c \"COPY $table FROM '$file_path' WITH (FORMAT csv, DELIMITER '|');\""
            
            # (選用) 匯入後刪除原始檔以節省空間，因為 SF=10 檔案很大
            rm "$file_path"
        fi
    done

    # 8. 建立必要的 Primary Keys / Foreign Keys (建議步驟)
    # 如果你有 dss.ri 檔案，可以在這裡執行
    # su - postgres -c "psql -d $DB_NAME -f $TPCH_DIR/dss.ri"

    # 9. 權限設定
    # 對應你的: GRANT SELECT ON ALL TABLES...
    echo "[Init] Granting permissions..."
    su - postgres -c "psql -d $DB_NAME -c \"GRANT SELECT ON ALL TABLES IN SCHEMA public TO $DB_USER;\""

    # 10. 優化資料庫
    # 對應你的: VACUUM ANALYZE
    echo "[Init] Running VACUUM ANALYZE..."
    su - postgres -c "psql -d $DB_NAME -c \"VACUUM ANALYZE;\""

    # 11. 停止暫時的 DB
    echo "[Init] Initialization complete. Stopping temp DB..."
    su - postgres -c "/usr/lib/postgresql/15/bin/pg_ctl -D $PG_DATA -m fast stop"
else
    echo "[Init] Database already initialized. Skipping setup."
fi

# --- 正式啟動階段 ---

echo "[System] Starting PostgreSQL Server..."
# 確保 Log 目錄權限
mkdir -p /var/log/postgresql
chown -R postgres:postgres /var/log/postgresql

# 背景啟動 DB
su - postgres -c "/usr/lib/postgresql/15/bin/pg_ctl -D $PG_DATA -l /var/log/postgresql/server.log start"

# 等待 DB Ready
echo "[System] Waiting for DB to be ready..."
sleep 3

echo "[System] Database is ready. Starting Application..."
# 切換回工作目錄
cd /app
# 執行 Docker CMD 傳入的指令 (python src/test_ppo.py ...)
exec "$@"