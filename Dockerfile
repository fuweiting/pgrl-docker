FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PGDATA=/var/lib/postgresql/data

# 1. Install necessary packages including PostgreSQL and development tools for dbgen
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    make \
    postgresql-15 \
    postgresql-client-15 \
    sudo \
    procps \
    && rm -rf /var/lib/apt/lists/*

# 2. Download and compile TPC-H dbgen during the build stage
WORKDIR /opt
RUN git clone https://github.com/gregrahn/tpch-kit.git tpch-dbgen \
    && cd tpch-dbgen/dbgen \
    && make \
    && cd /opt \
    && git clone https://github.com/tvondra/pg_tpch.git pg_tpch

# 3. Copy Python dependencies and code
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app

# 4. Copy entrypoint script
COPY entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

# 5. Set Entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Default command
CMD ["python", "src/test_ppo.py"]