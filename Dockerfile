FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PGDATA=/var/lib/postgresql/data

# PostgreSQL runs in the same container as the PGRL pipeline. Build tools are
# required for TPC-H dbgen.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    postgresql-15 \
    postgresql-client-15 \
    sudo \
    procps \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# TPC-H data generator and PostgreSQL TPC-H primary-key/index scripts.
WORKDIR /opt
RUN git clone --depth 1 https://github.com/gregrahn/tpch-kit.git tpch-dbgen \
    && cd /opt/tpch-dbgen/dbgen \
    && make \
    && git clone --depth 1 https://github.com/tvondra/pg_tpch.git /opt/pg_tpch

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src
COPY run_all.sh /app/run_all.sh
RUN chmod +x /app/run_all.sh \
    && mkdir -p /app/training_log/experiment_reports /app/merge_test_log /app/reference_configs

CMD ["bash", "/app/run_all.sh"]
