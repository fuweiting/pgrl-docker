FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PGDATA=/var/lib/postgresql/data

# 1. 安裝系統依賴 + PostgreSQL 15 + 編譯工具
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    make \
    postgresql-15 \
    postgresql-client-15 \
    sudo \
    procps \
    && rm -rf /var/lib/apt/lists/*

# 2. [優化] 在 Build 階段就先下載並編譯好 TPC-H dbgen
WORKDIR /opt
RUN git clone https://github.com/gregrahn/tpch-kit.git tpch-dbgen \
    && cd tpch-dbgen/dbgen \
    && make

# 3. 複製 Python 依賴與程式碼
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app

# 4. 複製啟動腳本
COPY entrypoint.sh /usr/local/bin/
# 安裝 dos2unix 來強制轉換換行格式
RUN apt-get update && apt-get install -y dos2unix && \
    dos2unix /usr/local/bin/entrypoint.sh && \
    chmod +x /usr/local/bin/entrypoint.sh && \
    # 清理 apt 快取減小容量
    rm -rf /var/lib/apt/lists/*

# 5. 設定 Entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# 預設指令
CMD ["python", "src/test_ppo.py"]