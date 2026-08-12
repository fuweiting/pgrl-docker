# Quick Start

This section describes the minimum configuration required to start the complete PGRL training, GCM, and validation pipeline in the local Docker TPC-H environment.

The default pipeline runs PostgreSQL, the TPC-H dataset, query-level PPO training, Global Configuration Merge (GCM), and final validation inside a single Docker container. Query training is performed sequentially; no per-query parallel containers are used.

## 1. Install Docker

Install Docker with Docker Compose support.

Verify the installation:

```bash
docker --version
docker compose version
```

## 2. Configure Environment Variables

Edit `.env` to configure the local PostgreSQL instance, TPC-H dataset, training workload, and GCM settings:

```dotenv
# Local PostgreSQL database created inside the PGRL container
TARGET_DB_NAME=tpch10
TARGET_DB_USER=pgrl
TARGET_DB_PASS=
TARGET_DB_PORT=5432

# PostgreSQL data/tuning paths inside the same container
PGDATA=/var/lib/postgresql/data
REMOTE_CONF_PATH=/var/lib/postgresql/data/auto_tuning.conf

# TPC-H dataset
TPCH_SCALE_FACTOR=10

# Query-level PPO training
P1_TIMESTEPS=2048
P2_TIMESTEPS=2048
P3_TIMESTEPS=2048

# Optional: override the workload, e.g. PGRL_QUERIES=Q1,Q2,Q3
PGRL_QUERIES=

# Regression-aware GCM
DEGRADATION_LAMBDA=2.5
DEGRADATION_POWER=2.0

# Fixed demo-environment parallel-worker limit
MAX_PARALLEL_WORKERS=20

# Used only while creating TPC-H indexes
LOAD_MAINTENANCE_WORK_MEM=2GB
```

Docker Compose loads these values into the PGRL container automatically.

`run_all.sh` exports the PostgreSQL connection through the standard libpq variables `PGDATABASE`, `PGUSER`, `PGPASSWORD`, `PGHOST`, and `PGPORT`. Training, GCM, and validation therefore use the same local PostgreSQL instance.

> `REMOTE_CONF_PATH` is retained as the environment-variable name for compatibility with the existing Python scripts. In this Docker environment, it points to the local `auto_tuning.conf` file inside the same container.

## 3. PostgreSQL and TPC-H Initialization

No external PostgreSQL server needs to be prepared manually.

When the container starts, `run_all.sh` checks the persistent PostgreSQL data directory and automatically performs first-time initialization when necessary.

On the first run, the script:

1. Initializes the PostgreSQL data directory with `initdb`.
2. Creates `auto_tuning.conf` and includes it from `postgresql.conf`.
3. Starts PostgreSQL locally inside the container.
4. Creates the configured PostgreSQL role and database.
5. Creates the TPC-H schema.
6. Generates TPC-H data using `dbgen`.
7. Loads the TPC-H tables.
8. Creates primary keys and indexes.
9. Runs `VACUUM ANALYZE`.
10. Records the initialized TPC-H scale factor in the PostgreSQL data volume.

The PostgreSQL data directory is stored in the Docker volume:

```text
pgrl_pg_data
```

After initialization has completed, later `docker compose up` executions reuse the existing PostgreSQL/TPC-H database instead of regenerating it.

If `TPCH_SCALE_FACTOR` is changed after the database has already been initialized, the existing volume must be removed before the dataset can be rebuilt.

## 4. PostgreSQL Tuning Configuration

The default tuning configuration path is:

```text
/var/lib/postgresql/data/auto_tuning.conf
```

The local PostgreSQL data directory therefore contains:

```text
/var/lib/postgresql/data/
├── postgresql.conf
├── auto_tuning.conf
└── ...
```

The main PostgreSQL configuration includes:

```conf
include = 'auto_tuning.conf'
```

`run_all.sh` ensures that the tuning file exists and that the include directive is present.

The same `auto_tuning.conf` file is used by:

- Query-level PPO training
- Global Configuration Merge
- Final validation

At the beginning of each pipeline run, `run_all.sh` clears `auto_tuning.conf` so that the underlying PostgreSQL/demo configuration is used as the system-default baseline instead of settings left by a previous interrupted run.

## 5. Configure the SQL Workload

The TPC-H workload is defined in:

```text
src/tpch_queryspecs.py
```

Queries are registered in `SQL_MAP` using `QuerySpec` objects.

For example:

```python
SQL_MAP = {
    "Q1": QuerySpec(
        name="Q1",
        sql="""
        SELECT ...
        """,
        params=(),
    ),
}
```

The default pipeline workload contains all TPC-H queries from `Q1` through `Q22`.

To use the complete workload, leave the following variable empty:

```dotenv
PGRL_QUERIES=
```

To run only selected queries, provide a comma-separated list. For example, to test only `Q1`, `Q2`, and `Q3`:

```dotenv
PGRL_QUERIES=Q1,Q2,Q3
```

The same workload is automatically passed to:

1. Query-level PPO training
2. Global Configuration Merge
3. Final configuration validation

Before training begins, `run_all.sh` verifies that every configured query exists in `SQL_MAP`.

## 6. Start the Pipeline

Build the Docker image and start the complete pipeline:

```bash
docker compose up --build
```

The pipeline executes:

```text
PostgreSQL / TPC-H Initialization
        │
        ▼
Sequential Query-Level PPO Training
        │
        ▼
Regression-Aware Global Configuration Merge
        │
        ▼
Final Configuration Validation
```

After the image has already been built, run:

```bash
docker compose up
```

To stop and remove the container while preserving the PostgreSQL volume:

```bash
docker compose down
```

To remove the container **and** the persistent PostgreSQL/TPC-H volume:

```bash
docker compose down -v
```

> `docker compose down -v` deletes the initialized PostgreSQL database. The complete TPC-H dataset will be generated again on the next startup.

## Next Steps

The Quick Start section covers the minimum requirements needed to launch the complete PGRL pipeline. Detailed descriptions of the PGRL architecture, training phases, configurable training parameters, regression-aware GCM, validation process, repository structure, and generated outputs are provided in the following sections.

&nbsp;

# PGRL: PostgreSQL Reinforcement Learning Tuning Framework

PGRL is a non-intrusive, hierarchical reinforcement learning framework for automatic PostgreSQL configuration tuning.

This Docker version provides a self-contained TPC-H environment in which PostgreSQL and PGRL execute inside the same container. The framework trains PostgreSQL configurations for individual SQL queries and subsequently uses a Global Configuration Merge mechanism to combine query-specific configurations into a workload-level configuration.

## Overview

Database configuration tuning usually requires significant DBA experience and extensive manual experimentation. PGRL automates this process by using reinforcement learning to explore PostgreSQL configuration parameters and identify configurations that improve query execution performance.

The framework consists of two major components:

1. **Query-Level Reinforcement Learning**

   Each SQL query is trained independently to identify high-performing PostgreSQL parameter values.

2. **Global Configuration Merge**

   Query-level configurations are merged into a single workload-level configuration while considering both overall performance and query regressions.

The automated Docker pipeline adds a final validation stage that compares the merged configuration with available reference configurations.

## Key Features

* Non-intrusive PostgreSQL tuning
* Query-aware reinforcement learning
* Hierarchical parameter exploration
* Sequential training for multiple SQL queries
* Regression-aware Global Configuration Merge
* Full-workload conflict evaluation
* Automated final configuration validation
* Self-contained PostgreSQL 15 and TPC-H environment
* Persistent PostgreSQL/TPC-H Docker volume
* Automatic first-run TPC-H generation and initialization
* Shared workload definition across training, GCM, and validation

## Tuning Workflow

The complete workflow contains three main PGRL stages after the local PostgreSQL/TPC-H environment has been prepared.

### Step 1: Query-Level Training

PGRL sequentially trains each configured SQL query using PPO.

Each query produces an experiment report containing the parameter values discovered during training.

```text
SQL Query
    │
    ▼
PPO Training
    │
    ▼
Query-Level Converged Parameters
```

Training is hierarchical and divided into three phases:

| Phase | Parameter Group | Application Mode |
| --- | --- | --- |
| Phase 1 | Planner and session-level parameters | Session |
| Phase 2 | JIT, GEQO, and query-planning parameters | Session |
| Phase 3 | Memory and I/O resource parameters | PostgreSQL restart |

### Step 2: Global Configuration Merge

The Global Configuration Merge mechanism collects the converged parameter values from all query-level training reports.

Parameters are classified into:

* **Non-conflicting parameters:** Parameters for which all converged queries agree on the same value. These values are directly applied to the base configuration.
* **Conflicting parameters:** Parameters for which different queries converge to different values. Candidate values are evaluated using regression-aware scoring across the complete workload.
* **Unconverged parameters:** Parameters that do not converge to a stable value during query-level training. These parameters retain their PostgreSQL system-default values.

Before evaluating conflicting parameter values, PGRL measures the latency of every workload query under the PostgreSQL system-default configuration. These measurements serve as the reference baselines for detecting query-level performance degradation.

Each conflicting candidate is then evaluated across the complete configured workload. This allows GCM to consider both overall performance and the effect of each candidate on queries that did not directly converge on the conflicting parameter.

Candidate values are progressively selected through regression-aware evaluation and global greedy selection, producing the final merged configuration `C_final`.

### Step 3: Final Validation

The final merged configuration is evaluated against the configured SQL workload and compared with the available reference configurations.

Multiple validation runs are performed to measure the stability and performance of each configuration. Per-query latency and total workload latency are reported for comparison.

## Regression-Aware GCM Scoring

PGRL uses a regression-aware scoring mechanism when evaluating conflicting parameter values.

For each query, the impact score considers:

* Normalized query latency
* Normalized query frequency
* Performance degradation relative to the system-default configuration

The impact score is defined as:

```text
Impact = (0.5 × S_latency + 0.5 × S_frequency)
         × (1 + λ × D^p)
```

where:

* `S_latency` is the normalized query latency
* `S_frequency` is the normalized execution frequency
* `D` is the degradation ratio relative to the system-default latency
* `λ` controls the strength of the degradation penalty
* `p` controls the degradation penalty exponent

The default values are:

```text
λ = 2.5
p = 2.0
```

Therefore, query degradation is penalized quadratically by default.

Before the merge process begins, PGRL evaluates every query in the configured workload under the system-default configuration. These per-query latency measurements are used as degradation baselines.

When evaluating conflicting parameters, PGRL executes the complete configured workload instead of evaluating only the queries that converged on the parameter. This prevents a parameter selection from causing hidden regressions in other queries.

## Repository Structure

```text
.
├── .env
├── .env.example
├── .gitattributes
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── README.md
├── requirements.txt
├── run_all.sh
├── reference_configs/
│   └── postgresql_default.conf
├── src/
│   ├── test_ppo.py
│   ├── pg_env.py
│   ├── merge_configs.py
│   ├── verify_configs.py
│   ├── tpch_queryspecs.py
│   └── param_specs.py
├── training_log/
│   └── experiment_reports/
└── merge_test_log/
```

### Main Files

| File | Description |
| --- | --- |
| `.env` | Stores local runtime settings such as the PostgreSQL database, TPC-H scale factor, training timesteps, selected workload, and GCM scoring values. This file is intended for machine-specific settings and should normally remain excluded from Git. |
| `.env.example` | Provides a version-controlled template for creating `.env`. |
| `.gitattributes` | Ensures shell scripts use LF line endings when the repository is used on Windows and Linux. |
| `.gitignore` | Excludes local environment files, generated logs, Python cache files, and other runtime artifacts that should not be committed. |
| `docker-compose.yml` | Defines the single PGRL Docker service, persistent PostgreSQL volume, shared-memory size, environment file, and output-directory mounts. |
| `Dockerfile` | Builds the runtime environment, installs PostgreSQL 15 and required build tools, builds TPC-H `dbgen`, installs Python dependencies, and starts the pipeline through `run_all.sh`. |
| `README.md` | Documents the PGRL architecture, local Docker setup, training workflow, GCM behavior, validation process, and outputs. |
| `requirements.txt` | Lists the Python packages installed into the Docker image. |
| `run_all.sh` | Initializes PostgreSQL/TPC-H when necessary and executes sequential query-level training, regression-aware GCM, and final validation. |
| `src/test_ppo.py` | Runs hierarchical PPO-based PostgreSQL parameter tuning and generates query-level experiment reports. |
| `src/pg_env.py` | Implements the PostgreSQL Gymnasium environment, parameter application, execution measurement, timeout handling, and PostgreSQL restart/reload operations. |
| `src/tpch_queryspecs.py` | Defines TPC-H `Q1` through `Q22` as `QuerySpec` objects. |
| `src/param_specs.py` | Defines PostgreSQL tuning parameters, value ranges, data types, and phase assignments. |
| `src/merge_configs.py` | Performs regression-aware Global Configuration Merge and produces the final workload-level configuration. |
| `src/verify_configs.py` | Evaluates reference configurations and the latest generated `C_final` against the selected workload. |

### Main Directories

| Directory | Description |
| --- | --- |
| `reference_configs/` | Stores reference PostgreSQL configurations used during final comparison. The default baseline is `postgresql_default.conf`. |
| `training_log/` | Stores per-query PPO logs and experiment outputs. |
| `training_log/experiment_reports/` | Stores consolidated query-level reports consumed by GCM. |
| `merge_test_log/` | Stores GCM evaluation logs, generated `C_final_*.conf` files, and related merge outputs. |

## SQL Workload

The TPC-H SQL workload is defined in:

```text
src/tpch_queryspecs.py
```

Each SQL statement is represented by a `QuerySpec` object containing:

* Query name
* SQL statement
* Optional query parameters

Example:

```python
QuerySpec(
    name="Q1",
    sql="""
    SELECT ...
    """,
    params=(),
)
```

The automated pipeline uses `PGRL_QUERIES` from `.env` to select the workload.

Use all TPC-H queries:

```dotenv
PGRL_QUERIES=
```

Use a subset:

```dotenv
PGRL_QUERIES=Q1,Q2,Q3
```

When the variable is empty, `run_all.sh` uses:

```text
Q1,Q2,Q3,...,Q22
```

The selected query list is validated against `SQL_MAP` before training begins and is then reused by training, GCM, and final validation.

## Requirements

The recommended execution environment requires:

* Docker
* Docker Compose

The Docker image provides the remaining runtime components, including:

* Python 3.10
* PostgreSQL 15
* PostgreSQL client tools
* TPC-H `dbgen`
* PostgreSQL TPC-H primary-key and index scripts
* Python dependencies from `requirements.txt`

Main Python dependencies include:

* `psycopg2-binary`
* `stable-baselines3`
* `gymnasium`
* `numpy`
* `pandas`
* `shimmy`

No separate PostgreSQL installation is required on the host for the default Docker pipeline.

## Environment Variables

The automated pipeline uses `.env` to configure the local PostgreSQL/TPC-H environment and PGRL execution settings.

```dotenv
# Local PostgreSQL database created inside the PGRL container
TARGET_DB_NAME=tpch10
TARGET_DB_USER=pgrl
TARGET_DB_PASS=
TARGET_DB_PORT=5432

# PostgreSQL data/tuning paths inside the same container
PGDATA=/var/lib/postgresql/data
REMOTE_CONF_PATH=/var/lib/postgresql/data/auto_tuning.conf

# TPC-H dataset
TPCH_SCALE_FACTOR=10

# Query-level PPO training
P1_TIMESTEPS=2048
P2_TIMESTEPS=2048
P3_TIMESTEPS=2048

# Optional workload override
PGRL_QUERIES=

# Regression-aware GCM
DEGRADATION_LAMBDA=2.5
DEGRADATION_POWER=2.0

# Fixed demo environment
MAX_PARALLEL_WORKERS=20
LOAD_MAINTENANCE_WORK_MEM=2GB
```

### Variable Descriptions

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `TARGET_DB_NAME` | No | `tpch10` | PostgreSQL database used by the TPC-H workload. |
| `TARGET_DB_USER` | No | `pgrl` | PostgreSQL role used by PGRL. The role is created automatically during first-time initialization if necessary. |
| `TARGET_DB_PASS` | No | Empty | PostgreSQL password value exported as `PGPASSWORD`. The default local initialization uses trusted local/container connections. |
| `TARGET_DB_PORT` | No | `5432` | PostgreSQL port used inside the container. |
| `PGDATA` | No | `/var/lib/postgresql/data` | PostgreSQL data directory backed by the persistent Docker volume. |
| `REMOTE_CONF_PATH` | No | `/var/lib/postgresql/data/auto_tuning.conf` | Tuning configuration file used by all pipeline stages. The legacy variable name is retained for script compatibility, but the file is local to the container. |
| `TPCH_SCALE_FACTOR` | No | `10` | TPC-H scale factor used when the dataset is first generated. |
| `P1_TIMESTEPS` | No | `2048` | PPO timesteps for Phase 1. |
| `P2_TIMESTEPS` | No | `2048` | PPO timesteps for Phase 2. |
| `P3_TIMESTEPS` | No | `2048` | PPO timesteps for Phase 3. |
| `PGRL_QUERIES` | No | Empty | Comma-separated workload override. Empty means `Q1` through `Q22`. |
| `DEGRADATION_LAMBDA` | No | `2.5` | Controls the strength of the query-regression penalty used by GCM. |
| `DEGRADATION_POWER` | No | `2.0` | Controls the exponent applied to the degradation ratio. |
| `MAX_PARALLEL_WORKERS` | No | `20` | Fixed demo-environment limit used to configure PostgreSQL parallel-worker capacity and the dynamic Phase-1 parallel levels. |
| `LOAD_MAINTENANCE_WORK_MEM` | No | `2GB` | Temporary `maintenance_work_mem` used only while creating TPC-H primary keys and indexes. It is not written as a PGRL tuned default. |

### Database Connection Handling

`run_all.sh` converts the database values into standard libpq environment variables:

```text
TARGET_DB_NAME → PGDATABASE
TARGET_DB_USER → PGUSER
TARGET_DB_PASS → PGPASSWORD
TARGET_DB_PORT → PGPORT
```

It also sets:

```text
PGHOST=127.0.0.1
```

The three pipeline stages therefore connect directly to the PostgreSQL instance running in the same container.

## Running the Complete Pipeline

The recommended execution method is Docker Compose:

```bash
docker compose up --build
```

`docker-compose.yml`:

1. Builds the runtime image.
2. Loads `.env`.
3. Allocates `4gb` of shared memory.
4. Mounts the persistent PostgreSQL volume.
5. Mounts the training, merge, and reference-configuration directories.
6. Starts `run_all.sh`.

`run_all.sh` performs the following operations:

1. Validates local database, path, and initialization settings.
2. Initializes PostgreSQL when the data volume is empty.
3. Initializes the TPC-H database when it has not been created yet.
4. Starts the local PostgreSQL server.
5. Validates the selected TPC-H query names.
6. Ensures `reference_configs/postgresql_default.conf` is available.
7. Sequentially trains every selected query.
8. Generates query-level experiment reports.
9. Runs regression-aware Global Configuration Merge.
10. Generates the final PostgreSQL configuration.
11. Runs final validation using the same workload.
12. Restores the underlying PostgreSQL configuration and stops PostgreSQL when the container exits.

After the image has been built, the pipeline can be started without rebuilding:

```bash
docker compose up
```

View container logs with:

```bash
docker compose logs -f
```

Stop and remove the container while preserving the database:

```bash
docker compose down
```

Reset the local PostgreSQL/TPC-H environment completely:

```bash
docker compose down -v
```

The `-v` option deletes `pgrl_pg_data`, so the next run performs a complete PostgreSQL and TPC-H initialization again.

## Training Configuration

### Training Arguments

| Argument | Default | Description |
| --- | ---: | --- |
| `--total-p1` | `2048` | Total PPO training timesteps for Phase 1. Set this value to `0` to skip planner/session-level tuning. |
| `--total-p2` | `2048` | Total PPO training timesteps for Phase 2. Set this value to `0` to skip JIT and GEQO tuning. |
| `--total-p3` | `2048` | Total PPO training timesteps for Phase 3. Set this value to `0` to skip memory and I/O tuning. |
| `--ent` | `0.01` | Initial PPO entropy coefficient. A higher value encourages broader exploration, while a lower value favors exploitation. |
| `--early-stop-factor` | `5.0` | Controls the dynamic catastrophic-latency threshold relative to the current baseline. |
| `--min-timeout-ms` | `3000` | Minimum SQL execution timeout in milliseconds. |
| `--timeout-penalty` | `-100.0` | Reward assigned when a training step times out or fails. |
| `--schedule` | `single` | Determines how queries are selected when more than one query is passed directly to `test_ppo.py`. |

The entropy coefficient is automatically annealed during each phase:

| Training Progress | Entropy Coefficient |
| --- | ---: |
| Beginning of the phase | Value specified by `--ent` |
| 50% of the phase | `0.001` |
| 80% of the phase | `0` |

This schedule encourages exploration during the early training stage and gradually shifts toward exploitation near the end of the phase.

### Workload and Output Parameters

| Argument | Default | Description |
| --- | --- | --- |
| `--queries` | `Q1` | Comma-separated SQL query names loaded from `src/tpch_queryspecs.py`. The automated pipeline invokes `test_ppo.py` once for each selected query. |
| `--schedule` | `single` | Query scheduling mode. The default pipeline uses `single` because queries are trained sequentially and independently. |
| `--report-dir` | `./training_log/experiment_reports` | Directory used to store consolidated query-level reports consumed by GCM. |

Supported scheduling modes are:

| Mode | Description |
| --- | --- |
| `single` | Uses a single query during training. This is the mode used by the default pipeline. |
| `round_robin` | Executes multiple supplied queries sequentially in a repeating order. |
| `random` | Randomly selects a query from the supplied workload at each step. |

### Local Database and Runtime Parameters

Database credentials are normally supplied through `.env` and exported by `run_all.sh`.

| Argument | Pipeline Value | Description |
| --- | --- | --- |
| `--dsn` | Application name only | Additional libpq options. Database name, user, password, host, and port are provided through the `PG*` environment variables. |
| `--remote-db-port` | `TARGET_DB_PORT` or `5432` | PostgreSQL port used by the local container. The argument name is retained for compatibility with the shared Python code. |
| `--remote-conf` | `REMOTE_CONF_PATH` | Path to the local `auto_tuning.conf` file. The argument name is retained for compatibility with the shared Python code. |

The default automated pipeline does not pass connection-management options because PostgreSQL runs locally in the same container.

## GCM Configuration

The Global Configuration Merge process is executed by `src/merge_configs.py`. It reads query-level training reports, classifies tuned parameters, evaluates conflicting values, and generates a final workload-level configuration.

`run_all.sh` passes the same local PostgreSQL port, tuning file, and workload used during training.

### GCM-Specific Arguments

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--report-dir` | String | `./training_log/experiment_reports` | Directory containing the query-level training reports generated by `test_ppo.py`. |
| `--queries` | String | `Q1,Q2,Q3` | Comma-separated query names included in the GCM workload. The automated pipeline overrides this value with the workload selected by `PGRL_QUERIES`. |
| `--degradation-lambda` | Float | `2.5` | Controls the strength of the regression penalty. The value must be greater than or equal to `0`. |
| `--degradation-power` | Float | `2.0` | Controls the exponent applied to the degradation ratio. The value must be greater than `0`. |

### Regression-Aware Scoring Parameters

The regression-aware impact score is defined as:

```text
Impact = (0.5 × S_latency + 0.5 × S_frequency)
         × (1 + λ × D^p)
```

| Argument | Formula Symbol | Valid Range | Effect |
| --- | --- | --- | --- |
| `--degradation-lambda` | `λ` | `λ ≥ 0` | Controls how strongly query degradation increases the impact score. A larger value makes GCM more conservative toward regressions. |
| `--degradation-power` | `p` | `p > 0` | Controls how rapidly the penalty grows as degradation increases. A value of `2.0` penalizes degradation quadratically. |

Examples of different scoring behaviors:

| Configuration | Behavior |
| --- | --- |
| `λ = 0` | Disables the degradation multiplier. The score only considers normalized latency and query frequency. |
| `λ = 2.5`, `p = 1.0` | Applies a linear degradation penalty. |
| `λ = 2.5`, `p = 2.0` | Applies the default quadratic degradation penalty. |
| Larger `λ` | Increases sensitivity to query-level regressions. |
| Larger `p` | Applies a rapidly increasing penalty to severe regressions. |

### Query Workload

The workload is supplied as a comma-separated list:

```bash
--queries "Q1,Q2,Q3"
```

Each query must satisfy both of the following conditions:

1. It must be defined in `src/tpch_queryspecs.py`.
2. Its training report must exist in the configured report directory.

The expected report filename is:

```text
<query-name>_experiment_report.log
```

For example:

```text
training_log/experiment_reports/Q1_experiment_report.log
training_log/experiment_reports/Q2_experiment_report.log
training_log/experiment_reports/Q3_experiment_report.log
```

The automated pipeline passes the same workload to query-level training, GCM, and final validation.

### Query Frequencies

Query execution frequencies can be specified in the `QUERY_FREQUENCIES` dictionary inside `src/merge_configs.py`:

```python
QUERY_FREQUENCIES = {
    "Q1": 10,
    "Q2": 5,
    "Q3": 20,
}
```

Queries that are not explicitly listed receive a default frequency of `1`.

The frequencies are normalized during scoring:

```text
S_frequency = query frequency / maximum workload frequency
```

Higher-frequency queries therefore receive greater importance in the regression-aware workload score. Query frequencies are currently configured in the source code and are not exposed as command-line arguments.

## Output Files

### Query-Level Training Logs

Detailed per-phase training logs are stored below:

```text
training_log/<query>/<timestamp>/
```

For example:

```text
training_log/Q1/20260808_102256/Q1_P1_steps2048.log
```

### Query-Level Experiment Reports

The consolidated query-level reports consumed by GCM are stored in:

```text
training_log/experiment_reports/
```

The expected report filename is:

```text
<query-name>_experiment_report.log
```

GCM reads the `[Final Consolidated Summary]` section of each report and extracts parameters marked as:

```text
STABLE
```

The extracted parameters are classified into:

| Category | Behavior |
| --- | --- |
| Non-conflicting | The converged value is added directly to the base configuration. |
| Conflicting | Candidate values are evaluated across the complete workload. |
| Unconverged | The PostgreSQL system-default value is retained. |

### Current Fixed GCM Settings

Some GCM settings remain defined directly in `merge_configs.py`.

| Setting | Current Value | Description |
| --- | ---: | --- |
| Per-query evaluation timeout | `30000 ms` | Maximum execution time assigned to each query during GCM evaluation. A timed-out query is recorded as `30000 ms`. |
| Local decision threshold | `2%` | Score differences within 2% of the current workload score activate the tie-breaker. |
| Latency weight | `0.5` | Weight assigned to normalized query latency in the impact score. |
| Frequency weight | `0.5` | Weight assigned to normalized query frequency in the impact score. |
| GCM final validation runs | `3` | Number of workload executions performed by GCM after `C_final` has been selected. |

Changing these settings requires modifying `src/merge_configs.py`.

### Final Merged Configuration

After the merge process completes, the final configuration is stored in:

```text
merge_test_log/C_final_<timestamp>.conf
```

The generated file contains:

* The GCM scoring profile
* The generation timestamp
* The total algorithm duration
* The final PostgreSQL parameter values

Example:

```text
# PGRL GCM Generated Configuration
# Scoring Profile: regression_aware
# Timestamp: 20260812_120000
# Algorithm Duration: 35.5 minutes

work_mem = '64MB'
enable_nestloop = 'off'
shared_buffers = '8192MB'
```

After GCM completes, the temporary tuning configuration is cleared and PostgreSQL is restored to the underlying local configuration before the pipeline proceeds.

## Final Validation Configuration

The final validation process is executed by `src/verify_configs.py`.

It compares:

1. PostgreSQL reference configurations stored in `reference_configs/`.
2. The latest `C_final_*.conf` generated by GCM.

The validation stage uses the same local PostgreSQL port, tuning file, and workload as training and GCM.

### Validation-Specific Arguments

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--ref-dir` | String | `./reference_configs` | Directory containing reference PostgreSQL configuration files. All `.conf` files in this directory are loaded automatically. |
| `--merged-dir` | String | `./merge_test_log` | Directory containing configurations generated by GCM. The validation script selects the most recently modified `C_final_*.conf`. |
| `--queries` | String | `Q1,Q2,Q3` | Workload used during validation. The automated pipeline overrides this with the selected `PGRL_QUERIES` workload. |
| `--remote-db-port` | Integer | `5432` | Local PostgreSQL port. The legacy argument name is retained for compatibility. |
| `--remote-conf` | String | `REMOTE_CONF_PATH` | Local `auto_tuning.conf` path. The legacy argument name is retained for compatibility. |

### Configuration Discovery

The validation script dynamically builds the test suite from two sources.

#### Reference Configurations

All `.conf` files in the directory specified by `--ref-dir` are loaded as reference configurations.

The repository includes:

```text
reference_configs/
└── postgresql_default.conf
```

`postgresql_default.conf` is intentionally empty except for comments:

```conf
# Empty by design.
# Applying this file clears auto_tuning.conf and evaluates the underlying
# PostgreSQL/demo configuration as the validation baseline.
```

Applying this configuration clears `auto_tuning.conf`, so validation measures the underlying PostgreSQL/demo configuration as the default baseline.

Additional reference configurations may be added to the same directory.

#### PGRL Configuration

The script searches the directory specified by `--merged-dir` for files matching:

```text
C_final_*.conf
```

When multiple generated configurations are available, only the most recently modified file is selected for validation.

For example:

```text
merge_test_log/
├── C_final_20260811_153000.conf
└── C_final_20260812_103000.conf
```

In this example, `C_final_20260812_103000.conf` is selected.

### Configuration File Format

The validation script accepts PostgreSQL configuration entries written with an equals sign:

```text
work_mem = '64MB'
enable_nestloop = 'off'
shared_buffers = '8192MB'
```

It can also parse entries written with a colon or whitespace separator:

```text
work_mem: 64MB
enable_nestloop off
```

Comments beginning with `#` are ignored.

### Validation Procedure

For each discovered configuration, the validation script performs the following steps:

1. Writes the configuration to the local `auto_tuning.conf`.
2. Restarts the local PostgreSQL server using `pg_ctl`.
3. Executes the complete target workload once as a warm-up.
4. Executes the workload five additional times for measurement.
5. Calculates the average total latency and average per-query latency from the five recorded runs.
6. Compares the results with the default reference configuration.

The warm-up run is excluded from the reported average.

```text
Total executions per configuration: 6
Warm-up executions:               1
Recorded executions:              5
```

### Validation Timeout

Each SQL query has a maximum execution time of:

```text
300000 ms
```

This is equivalent to five minutes per query.

When a query times out or encounters an execution error, its latency is recorded as `300000 ms` for that run.

The validation timeout is currently defined directly in `src/verify_configs.py` and is not exposed as a command-line argument.

> The validation timeout differs from the GCM evaluation timeout. GCM uses a shorter timeout for candidate exploration, while final validation allows a longer execution period to measure slow reference configurations more completely.

### Baseline Comparison

After evaluating all configurations, `verify_configs.py` searches for a successfully evaluated configuration whose name contains:

```text
default
```

The included `postgresql_default.conf` therefore becomes the validation baseline.

The improvement percentage is calculated as:

```text
Improvement (%) =
    (Baseline Latency - Configuration Latency)
    / Baseline Latency
    × 100
```

A positive value indicates that the evaluated configuration is faster than the baseline. A negative value indicates a performance regression.

### Example

With PostgreSQL already running inside the PGRL container and the required `PG*` variables exported, validation can be executed with:

```bash
python src/verify_configs.py \
  --dsn "application_name=PGRL-VERIFY-DEMO" \
  --remote-db-port "${TARGET_DB_PORT:-5432}" \
  --remote-conf "${REMOTE_CONF_PATH}" \
  --ref-dir "/app/reference_configs" \
  --merged-dir "/app/merge_test_log" \
  --queries "Q1,Q2,Q3"
```

Under the normal workflow, users do not need to run this command manually because `run_all.sh` executes validation automatically.

### Validation Settings in `run_all.sh`

The automated pipeline invokes validation using the same workload and local PostgreSQL settings:

```bash
python src/verify_configs.py \
    --dsn "application_name=PGRL-VERIFY-DEMO" \
    --remote-db-port "$DB_PORT" \
    --remote-conf "$TUNING_CONF" \
    --ref-dir "$REFERENCE_DIR" \
    --merged-dir "$MERGE_DIR" \
    --queries "$QUERIES_STR"
```

`QUERIES_STR` is generated from `PGRL_QUERIES`, or from the default `Q1` through `Q22` workload when `PGRL_QUERIES` is empty.

### Current Fixed Validation Settings

| Setting | Current Value | Description |
| --- | ---: | --- |
| Query timeout | `300000 ms` | Maximum execution time for each SQL query during validation. |
| Total runs per configuration | `6` | Number of workload executions performed for each configuration. |
| Warm-up runs | `1` | First execution used to warm the database and caches. |
| Recorded runs | `5` | Executions included in the reported average. |

### Cleanup

After all configurations have been evaluated, the validation script:

1. Clears `auto_tuning.conf`.
2. Restarts the local PostgreSQL server.
3. Restores the underlying PostgreSQL/demo configuration.

When the complete pipeline exits, `run_all.sh` also stops PostgreSQL cleanly.

## Security Notes

Do not commit machine-specific passwords or other sensitive runtime values.

Store environment-specific values in `.env` and keep `.env` excluded from Git. Commit `.env.example` instead when a reusable configuration template is needed.

Recommended `.gitignore` entries include:

```gitignore
.env

training_log/
merge_test_log/

__pycache__/
*.py[cod]

.venv/
venv/

.vscode/
.idea/
.DS_Store
Thumbs.db
```

The persistent PostgreSQL data is stored in a Docker named volume (`pgrl_pg_data`) rather than in the repository and should not be committed.

Because `run_all.sh` exports `TARGET_DB_PASS` as `PGPASSWORD`, the password is available inside the container environment when one is configured. Access to the host and Docker daemon should therefore remain appropriately restricted.
