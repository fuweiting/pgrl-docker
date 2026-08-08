# verify_configs_v6.py

import argparse
import psycopg2
import time
import sys
import os
import glob
import subprocess
import shlex
import paramiko
from pathlib import Path
from sshtunnel import SSHTunnelForwarder
from typing import Dict, Any, List, Tuple

try:
    from tpch_queryspecs import SQL_MAP
except ImportError:
    print("Error: Failed to load 'tpch_queryspecs.py'.", file=sys.stderr)
    sys.exit(1) 

# Parse .conf files into dictionary
def parse_pg_conf(filepath: Path) -> Dict[str, str]:
    params = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.split('#')[0].strip()
                if not line:
                    continue
                
                line = line.rstrip(',')
                
                if '=' in line:
                    k, v = line.split('=', 1)
                elif ':' in line:
                    k, v = line.split(':', 1)
                else:
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2:
                        k, v = parts
                    else:
                        continue
                        
                k = k.strip().strip("'").strip('"')
                v = v.strip().strip("'").strip('"')
                params[k] = v
                
        print(f"      [Debug] Parsed {len(params)} parameters from {filepath.name}")
        
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return params

# Dynamically builds the TEST_SUITE from reference and generated config files.
def build_test_suite(ref_dir: str, merged_dir: str) -> List[Dict[str, Any]]:
    test_suite = []
    
    # 1. Load static reference configs (e.g., sunbird_default, cybertec)
    ref_path = Path(ref_dir)
    if ref_path.exists():
        for conf_file in ref_path.glob("*.conf"):
            test_suite.append({
                "name": conf_file.stem,
                "description": f"Reference Config: {conf_file.name}",
                "params": parse_pg_conf(conf_file)
            })
            print(f"[System] Loaded reference config: {conf_file.name}")
    else:
        print(f"[Warning] Reference directory '{ref_dir}' not found.")

    # 2. Automatically find the LATEST merged config from PGRL
    merged_path = Path(merged_dir)
    if merged_path.exists():
        # Find all C_final_*.conf files and sort them by modification time
        final_confs = list(merged_path.glob("C_final_*.conf"))
        if final_confs:
            latest_conf = max(final_confs, key=os.path.getmtime)
            test_suite.append({
                "name": latest_conf.stem,
                "description": "LATEST PGRL GCM Generated Configuration",
                "params": parse_pg_conf(latest_conf)
            })
            print(f"[System] Loaded latest PGRL merged config: {latest_conf.name}")
        else:
            print(f"[Warning] No PGRL generated 'C_final_*.conf' found in '{merged_dir}'.")
    
    return test_suite

class ConfigTester:
    def __init__(self, dsn, ssh_client, ssh_password, target_queries, remote_conf_path):
        self.dsn = dsn
        self.ssh_client = ssh_client
        self.ssh_password = ssh_password
        self.workload_q_map = SQL_MAP
        self.target_queries = target_queries
        self.remote_conf_path = remote_conf_path
        print(f"ConfigTester initialized. PostgreSQL tuning config file: {self.remote_conf_path}")
        
    def _run_workload_once(self) -> Tuple[float, Dict[str, float]]:
        total_latency = 0.0
        query_latencies = {} 
        EVAL_TIMEOUT_MS = 300_000 
        conn = None

        try:
            for attempt in range(10):
                try:
                    conn = psycopg2.connect(self.dsn)
                    conn.autocommit = True
                    break
                except psycopg2.OperationalError:
                    time.sleep(2.0)

            if not conn:
                print("        -> Unable to connect to the database.")
                return -1.0, {}

            with conn.cursor() as cur:
                cur.execute(f"SET statement_timeout = {EVAL_TIMEOUT_MS}")

                for q_name in self.target_queries:
                    if q_name not in self.workload_q_map:
                        continue

                    q_spec = self.workload_q_map[q_name]
                    t0 = time.perf_counter()
                    try:
                        cur.execute(q_spec.sql, q_spec.params)
                        cur.fetchall()
                        lat = (time.perf_counter() - t0) * 1000.0
                        total_latency += lat
                        query_latencies[q_name] = lat 
                    except Exception as e:
                        print(f"        -> {q_name}: TIMEOUT/ERROR ({e})")
                        total_latency += EVAL_TIMEOUT_MS
                        query_latencies[q_name] = float(EVAL_TIMEOUT_MS)

        finally:
            if conn: conn.close()
        return total_latency, query_latencies

    def _local_pgdata(self) -> str:
        return os.getenv("PGDATA") or os.path.dirname(self.remote_conf_path)

    def _local_pg_ctl(self) -> str:
        explicit = os.getenv("PG_CTL_PATH")
        if explicit:
            return explicit
        candidates = glob.glob("/usr/lib/postgresql/*/bin/pg_ctl")
        if candidates:
            def version_key(path: str):
                version = path.split("/postgresql/", 1)[1].split("/", 1)[0]
                try:
                    return tuple(int(part) for part in version.split("."))
                except ValueError:
                    return (0,)
            return max(candidates, key=version_key)
        return "pg_ctl"

    def _restart_remote_postgresql(self) -> bool:
        """Restart PostgreSQL in local Docker mode or on a remote SSH host."""
        if self.ssh_client is None:
            try:
                pg_ctl = self._local_pg_ctl()
                pgdata = self._local_pgdata()
                command = f"{shlex.quote(pg_ctl)} -D {shlex.quote(pgdata)} -w restart"
                print("      [System] Restarting local PostgreSQL...")
                subprocess.run(["su", "-", "postgres", "-c", command], check=True)
                time.sleep(2.0)
                return True
            except Exception as e:
                print(f"      [Error] Local PostgreSQL restart failed: {e}")
                return False

        sudo_flag = "-S" if self.ssh_password else "-n"
        cmd = f"sudo {sudo_flag} systemctl restart postgresql"
        print("      [System] Restarting remote PostgreSQL...")
        try:
            stdin, stdout, stderr = self.ssh_client.exec_command(cmd, get_pty=True)
            if self.ssh_password:
                stdin.write(self.ssh_password + "\n")
                stdin.flush()
            if stdout.channel.recv_exit_status() != 0:
                print(f"      [Error] Restart failed: {stderr.read().decode()}")
                return False
            time.sleep(8.0)
            return True
        except Exception as e:
            print(f"      [Error] SSH execution exception: {e}")
            return False

    def _update_remote_config(self, params: Dict[str, str]) -> bool:
        config_lines = [f"{key} = '{value}'" for key, value in params.items()]
        config_content = "\n".join(config_lines)
        target_path = self.remote_conf_path

        if self.ssh_client is None:
            try:
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(config_content)
                subprocess.run(["chown", "postgres:postgres", target_path], check=True)
                subprocess.run(["chmod", "644", target_path], check=True)
                return True
            except Exception as e:
                print(f"      [Error] Local config write failed: {e}")
                return False

        temp_path = "/tmp/auto_tuning.tmp"
        try:
            sftp = self.ssh_client.open_sftp()
            with sftp.file(temp_path, "w") as f:
                f.write(config_content)
            sftp.close()

            sudo_flag = "-S" if self.ssh_password else "-n"
            mv_cmd = (
                f"sudo {sudo_flag} mv -f {temp_path} {target_path} && "
                f"sudo {sudo_flag} chown postgres:postgres {target_path} && "
                f"sudo {sudo_flag} chmod 644 {target_path}"
            )
            stdin, stdout, stderr = self.ssh_client.exec_command(mv_cmd, get_pty=True)
            if self.ssh_password:
                stdin.write(self.ssh_password + "\n")
                stdin.flush()
            if stdout.channel.recv_exit_status() != 0:
                print(f"      [Error] Failed to update Config: {stderr.read().decode()}")
                return False
            return True
        except Exception as e:
            print(f"      [Error] SSH write exception: {e}")
            return False

    def restore_defaults(self):
        print("\n" + "="*40)
        print("Restoring PostgreSQL settings (underlying/default configuration)...")
        if self._update_remote_config({}): 
            print("      [System] Config file cleared (auto_tuning.conf)")
            if self._restart_remote_postgresql():
                print("      [System] PostgreSQL restarted and restored")
            else:
                print("      [Warning] Restore restart failed")
        else:
            print("      [Warning] Failed to clear config file")
        print("="*40 + "\n")

    def evaluate(self, config_name: str, params: Dict[str, str]) -> Tuple[float, Dict[str, float]]:
        print(f"\n--- Starting evaluation: {config_name} ---")
        if not self._update_remote_config(params): return -1.0, {}
        if not self._restart_remote_postgresql():
            print("Failed to restart DB, skipping this test.")
            return -1.0, {}

        EVAL_REPEAT = 6 
        latencies: List[float] = []
        query_history: Dict[str, List[float]] = {q: [] for q in self.target_queries}

        print(f"      [Running] Executing {EVAL_REPEAT} times (1 warmup + 5 recorded) per config to get average")

        for i in range(EVAL_REPEAT):
            if i == 0:
                print(f"        Round {i + 1}/{EVAL_REPEAT} [Warmup] ...", end="")
            else:
                print(f"        Round {i + 1}/{EVAL_REPEAT} ..........", end="")
                
            lat, q_lats = self._run_workload_once()

            if lat < 0:
                print("        This round failed, marking this config as FAILED")
                return -1.0, {}
            print(f" Total time: {lat:.2f} ms")
            for q, l in q_lats.items():
                print(f"          - {q}: {l:.2f} ms")
            
            if i > 0:
                latencies.append(lat)
                for q, l in q_lats.items():
                    query_history[q].append(l)

        avg_total_latency = sum(latencies) / len(latencies)
        avg_query_latencies = {q: sum(times) / len(times) for q, times in query_history.items() if times}

        print(f"      [Result] Average Total Latency: {avg_total_latency:.2f} ms")
        return avg_total_latency, avg_query_latencies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dsn",
        default=os.getenv("PGRL_DSN", "application_name=PGRL-VERIFY"),
        help=(
            "Base DSN without the SSH-forwarded host/port. Database name, user, "
            "and password may also be supplied through PGDATABASE, PGUSER, and "
            "PGPASSWORD."
        ),
    )
    ap.add_argument("--ssh-host", default=os.getenv("TARGET_SSH_HOST", ""))
    ap.add_argument("--ssh-port", type=int, default=int(os.getenv("TARGET_SSH_PORT", "22")))
    ap.add_argument("--ssh-user", default=os.getenv("TARGET_SSH_USER", ""))
    ap.add_argument("--ssh-key", default=os.getenv("TARGET_SSH_KEY") or None)
    ap.add_argument("--ssh-password", default=os.getenv("TARGET_SSH_PASS", ""))
    ap.add_argument("--local-port", type=int, default=int(os.getenv("LOCAL_FORWARD_PORT", "5433")))
    ap.add_argument(
        "--remote-db-port",
        type=int,
        default=int(os.getenv("TARGET_DB_PORT", "5432")),
        help="PostgreSQL port on the remote server.",
    )
    ap.add_argument(
        "--remote-conf",
        default=os.getenv(
            "REMOTE_CONF_PATH",
            os.path.join(os.getenv("PGDATA", "/var/lib/pgsql/data"), "auto_tuning.conf"),
        ),
        help="Remote auto_tuning.conf path used by all pipeline stages.",
    )

    ap.add_argument("--ref-dir", default="./reference_configs", help="Directory containing baseline .conf files")
    ap.add_argument("--merged-dir", default="./merge_test_log", help="Directory containing PGRL generated C_final_*.conf files")
    
    ap.add_argument("--queries", default="Q1,Q2,Q3", help="Comma-separated list of queries to process")
    args = ap.parse_args()
    
    target_queries = [q.strip() for q in args.queries.split(',') if q.strip()]

    # Dynamically build the test suite
    print("Scanning for configurations...")
    TEST_SUITE = build_test_suite(args.ref_dir, args.merged_dir)
    
    if not TEST_SUITE:
        print("Error: No configurations found to test. Exiting.")
        sys.exit(1)

    remote_mode = bool(args.ssh_host and args.ssh_host not in {"localhost", "127.0.0.1"})
    forwarder = None
    ssh_client = None
    tester = None

    try:
        if remote_mode:
            print("Connecting to SSH...")
            forwarder = SSHTunnelForwarder(
                (args.ssh_host, args.ssh_port),
                ssh_username=args.ssh_user,
                ssh_password=args.ssh_password or None,
                ssh_pkey=args.ssh_key,
                remote_bind_address=("127.0.0.1", args.remote_db_port),
                local_bind_address=("127.0.0.1", args.local_port),
                set_keepalive=30.0,
            )
            forwarder.start()

            ssh_client = paramiko.SSHClient()
            ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh_client.connect(
                args.ssh_host,
                port=args.ssh_port,
                username=args.ssh_user,
                password=args.ssh_password or None,
                key_filename=args.ssh_key,
            )
            forwarded_dsn = f"{args.dsn} host=127.0.0.1 port={forwarder.local_bind_port} connect_timeout=10 sslmode=disable"
        else:
            print("[System] Running validation in Local Mode (No SSH Tunnel).")
            forwarded_dsn = f"{args.dsn} host=127.0.0.1 port={args.remote_db_port} connect_timeout=10 sslmode=disable"

        tester = ConfigTester(
            forwarded_dsn,
            ssh_client,
            args.ssh_password,
            target_queries,
            args.remote_conf,
        )

        print(f"\nPreparing to test {len(TEST_SUITE)} configurations...\n")

        results = []
        for test_case in TEST_SUITE:
            name = test_case['name']
            params = test_case['params']
            latency, q_lats = tester.evaluate(name, params)
            results.append((name, latency, q_lats))

        # Print final results summary
        print("\n" + "="*60)
        print("Test Results Summary (Average over runs)")
        print("="*60)
        
        baseline_lat = None
        baseline_name = None
        # Prefer a reference configuration whose name contains "default".
        for name, latency, _ in results:
            if "default" in name.lower() and latency >= 0:
                baseline_lat = latency
                baseline_name = name
                break
            
        for name, latency, q_lats in results:
            if latency >= 0:
                print(f"[{name}] Total time: {latency:.2f} ms")
                for q in target_queries:
                    if q in q_lats:
                        print(f"  └─ {q}: {q_lats[q]:.2f} ms")
            else:
                print(f"[{name}]: FAILED")
            print("-" * 30)
            
        if baseline_lat and baseline_lat > 0:
            print(f"\n--- Improvement (Compared to {baseline_name}) ---")
            for name, latency, _ in results:
                if name != baseline_name and latency > 0:
                    imp = ((baseline_lat - latency) / baseline_lat) * 100
                    print(f"{name:<30}: {imp:+.2f}%")
        else:
            print("\n[Warning] No reference configuration containing 'default' was found; improvement percentages were skipped.")
        print("="*60)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if tester: tester.restore_defaults()
        if ssh_client: ssh_client.close()
        if forwarder and forwarder.is_active: forwarder.stop()

if __name__ == "__main__":
    main()