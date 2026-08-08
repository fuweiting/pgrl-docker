# test_ppo_v14.py
# 3-Phase Hierarchical Training
# Phase 1: Planner (Session)
# Phase 2: JIT & GEQO (Session) 
# Phase 3: Memory & I/O Resources (Restart)
# Added Convergence Stopping Mechanism for P2/P3 phases
# Added Dynamic Parallel Degree Parameter Mapping
# Fixed extract_converged_params_from_log() to excluded cache-influenced steps
# Added ssh_host check for local mode
# Modified parallel degree handling: removed virtual parameter and directly tuned max_parallel_workers_per_gather with dynamic levels
# Added catastrophic latency early stopping in callback
# Added pre-flight check to ensure include directive exists before starting SSH tunnel

from pg_env import PgConfEnv, PPOLogger, ConvergenceStoppingCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import contextlib, argparse
import os
import re
import paramiko 
import psycopg2
import time
from sshtunnel import SSHTunnelForwarder
from pathlib import Path
from datetime import datetime
from collections import Counter

from tpch_queryspecs import SQL_MAP
from param_specs import P1_PARAM_SPECS, P2_PARAM_SPECS, P3_PARAM_SPECS

DEFAULT_TOTAL_P1 = 2048
DEFAULT_TOTAL_P2 = 2048
DEFAULT_TOTAL_P3 = 2048
DEFAULT_THRESHOLD = 0.8
TEST_SQL = "Q1"

# This function resolves virtual parameters in the config by mapping them to their corresponding real parameters based on the provided specs.
def resolve_virtual_params(config: dict, specs: dict) -> dict:
    new_config = {}
    for k, v in config.items():
        if k not in specs:
            new_config[k] = v
            continue
        
        spec = specs[k]
        if spec.get("is_virtual", False):
            for target_param in spec.get("map_to", []):
                new_config[target_param] = v
        else:
            new_config[k] = v
            
    return new_config

def extract_converged_params_from_log(log_path: Path, param_specs: dict, last_n: int = 20, threshold_ratio: float = DEFAULT_THRESHOLD, latency_tolerance: float = 0.2) -> tuple[dict, list]:
    report_lines = []
    def log_print(msg: str):
        print(msg)
        report_lines.append(msg)

    log_print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Analysis Target: {log_path.name}")
    log_print(f"Settings: Last {last_n} steps, Convergence Threshold {int(threshold_ratio * 100)}%")
    log_print(f"Filter: Only analyze steps within Top {int(latency_tolerance * 100)}% of best latency in window")
    log_print("-" * 40)

    if not log_path.exists():
        log_print("[Error] Log file not found!")
        return {}, report_lines

    step_data = []
    try:
        lat_regex = re.compile(r"latency_ms=([\d\.]+)")
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip().startswith("[") and "latency_ms=" in line:
                    match = lat_regex.search(line)
                    if match:
                        lat = float(match.group(1))
                        step_data.append((lat, line))
    except Exception as e:
        log_print(f"[Error] Failed to read file: {e}")
        return {}, report_lines
    
    if not step_data:
        log_print("[Analysis] No valid step lines found.")
        return {}, report_lines

    window_data = step_data[-last_n:]
    if len(window_data) < last_n:
        log_print(f"[Warning] Not enough steps ({len(window_data)} < {last_n}) for convergence check.")
    
    # Convergence analysis strategy:
    # 1. Sort the latencies in the window and find the 20th percentile (P20) latency to serve as a baseline. 
    # This helps to exclude the influence of the fastest 20% of steps, which are likely cache hits and not representative of the true execution plan performance.
    sorted_lats = sorted([d[0] for d in window_data])
    
    # 2. Use the P20 latency as the baseline for filtering.
    # This means we will only consider steps that are within a certain percentage (latency_tolerance) of this baseline latency, effectively excluding those that are significantly faster (likely due to caching) and focusing on those that reflect the actual execution plan performance.
    p_index = int(len(sorted_lats) * 0.20) 
    baseline_lat = sorted_lats[p_index]
    
    # 3. Calculate the cutoff latency based on the baseline and the specified tolerance. Only steps with latency less than or equal to this cutoff will be considered for convergence analysis.
    cutoff_lat = baseline_lat * (1.0 + latency_tolerance)
    
    # 4. Filter the steps based on the cutoff latency.
    filtered_lines = [line for lat, line in window_data if lat <= cutoff_lat]

    log_print(f"[Filter Stats] Baseline (P20): {baseline_lat:.2f} ms | Cutoff: {cutoff_lat:.2f} ms")
    log_print(f"[Filter Stats] Kept {len(filtered_lines)}/{len(window_data)} steps for analysis.")
    
    if len(filtered_lines) == 0:
        log_print("[Error] No steps passed the latency filter? This shouldn't happen.")
        return {}, report_lines

    converged_config = {}
    effective_n = len(filtered_lines)
    threshold = effective_n * threshold_ratio
    min_samples_required = max(3, int(last_n * 0.2)) 
    
    if effective_n < min_samples_required:
        log_print(f"[Warning] Too few high-quality steps ({effective_n}) to determine convergence. Skipping.")
        return {}, report_lines

    for param in param_specs.keys():
        regex = re.compile(rf"\b{param}=([^,\s]+)")
        values = []
        for line in filtered_lines:
            match = regex.search(line)
            if match:
                values.append(match.group(1).strip())
        
        if not values:
            log_print(f"  ? {param}: Not found.")
            continue

        counts = Counter(values)
        most_common_val, frequency = counts.most_common(1)[0]
        
        spec = param_specs[param]
        display_name = param
        if spec.get("is_virtual", False) and "map_to" in spec:
            display_name = spec["map_to"][0]

        status_msg = f"  -> {display_name}: "
        stats_msg = f"Mode={most_common_val} (count={frequency}/{effective_n})"
        
        if frequency >= threshold:
            converged_config[param] = most_common_val
            log_print(f"{status_msg}STABLE.   {stats_msg}")
        else:
            log_print(f"{status_msg}UNSTABLE. {stats_msg}")

    log_print("-" * 40)
    log_print(f"Converged params count: {len(converged_config)}/{len(param_specs)}")
    log_print("\n") 
    return converged_config, report_lines

def append_to_master_log(file_path: Path, lines: list):
    try:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except Exception as e:
        print(f"[System] Failed to write to master log: {e}")

def create_ssh_client(args):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        args.ssh_host, port=args.ssh_port, username=args.ssh_user, 
        password=args.ssh_password, key_filename=args.ssh_key
    )
    client.get_transport().set_keepalive(60)
    return client

# Auto-inject include directive into postgresql.conf
def ensure_include_directive(args):
    if not args.ssh_host or args.ssh_host in ["", "localhost", "127.0.0.1"]:
        return

    print("\n[System] Checking if 'auto_tuning.conf' is included in main postgresql.conf...")
    client = create_ssh_client(args)
    
    # Derive postgresql.conf path dynamically from --remote-conf
    main_conf_dir = os.path.dirname(args.remote_conf)
    main_conf_path = f"{main_conf_dir}/postgresql.conf"
    target_conf_name = os.path.basename(args.remote_conf)
    include_line = f"include = '{target_conf_name}'"
    
    try:
        # Check if the uncommented include line already exists
        check_cmd = f"sudo -S grep -q \"^{include_line}\" {main_conf_path}"
        stdin, stdout, stderr = client.exec_command(check_cmd)
        stdin.write(args.ssh_password + '\n')
        stdin.flush()
        
        if stdout.channel.recv_exit_status() == 0:
            print(f"      -> Include directive already present in {main_conf_path}.")
        else:
            print(f"      -> Include directive missing. Adding to {main_conf_path}...")
            
            # Append the line and restart
            append_cmd = f"echo \"{include_line}\" | sudo -S tee -a {main_conf_path} > /dev/null"
            stdin, stdout, stderr = client.exec_command(append_cmd)
            stdin.write(args.ssh_password + '\n')
            stdin.flush()
            
            if stdout.channel.recv_exit_status() == 0:
                print("      -> Successfully appended. Restarting PostgreSQL to apply changes...")
                restart_cmd = "sudo -S systemctl restart postgresql"
                stdin, stdout, stderr = client.exec_command(restart_cmd)
                stdin.write(args.ssh_password + '\n')
                stdin.flush()
                
                if stdout.channel.recv_exit_status() == 0:
                    print("      -> PostgreSQL restarted successfully.")
                    time.sleep(5)  # Buffer to let the DB recover completely
                else:
                    print(f"      [Error] Failed to restart DB: {stderr.read().decode()}")
            else:
                print(f"      [Error] Failed to append include directive: {stderr.read().decode()}")
                
    except Exception as e:
        print(f"      [Error] SSH execution failed during include check: {e}")
    finally:
        client.close()

class StepAnnealCB(BaseCallback):
    def __init__(self, total, mid=0.5, late=0.8, ent_mid=1e-3, ent_late=0, lr_mid=3e-4, lr_late=3e-4):
        super().__init__()
        self.total = total
        self.t_mid = int(total * mid)
        self.t_late = int(total * late)
        self.ent_mid = float(ent_mid)
        self.ent_late = float(ent_late)
        self.lr_mid = float(lr_mid)
        self.lr_late = float(lr_late)
        self.did_mid = False
        self.did_late = False

    def _on_step(self) -> bool:
        t = self.num_timesteps
        if (not self.did_mid) and t >= self.t_mid:
            self.model.ent_coef = self.ent_mid
            self.model.lr_schedule = lambda _: self.lr_mid
            print(f"[anneal] t={t} ent_coef -> {self.ent_mid}, lr -> {self.lr_mid}")
            self.did_mid = True
        if (not self.did_late) and t >= self.t_late:
            self.model.ent_coef = self.ent_late
            self.model.lr_schedule = lambda _: self.lr_late
            print(f"[anneal] t={t} ent_coef -> {self.ent_late}, lr -> {self.lr_late}")
            self.did_late = True
        return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dsn",
        default=os.getenv("PGRL_DSN", ""),
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

    ap.add_argument("--queries", default=TEST_SQL)
    ap.add_argument("--schedule", default="single", choices=["single","round_robin","random"])
    
    ap.add_argument("--total-p1", type=int, default=DEFAULT_TOTAL_P1)
    ap.add_argument("--total-p2", type=int, default=DEFAULT_TOTAL_P2)
    ap.add_argument("--total-p3", type=int, default=DEFAULT_TOTAL_P3)
    
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--early-stop-factor", type=float, default=5.0)
    ap.add_argument("--min-timeout-ms", type=int, default=3000)
    ap.add_argument("--timeout-penalty", type=float, default=-100.0)
    
    ap.add_argument("--report-dir", default="./training_log/experiment_reports", help="Directory to save the final summary report for GCM")
    
    args = ap.parse_args()
    
    ssh_ctrl = None
    forwarder = None
    
    qs = [SQL_MAP[name.strip()] for name in args.queries.split(',') if name.strip()]

    # Pre-Flight Check: Ensure include directive exists before starting tunnel
    ensure_include_directive(args)
    
    # Launch SSH Tunnel if needed, and construct the forwarded DSN for database connections. 
    # This allows the script to connect to a remote PostgreSQL instance securely, while still treating it as if it were local.
    if args.ssh_host and args.ssh_host not in ["", "localhost", "127.0.0.1"]:
        print(f"[System] Starting SSH Tunnel to {args.ssh_host}...")
        forwarder = SSHTunnelForwarder(
            (args.ssh_host, args.ssh_port),
            ssh_username=args.ssh_user,
            ssh_password=args.ssh_password,
            ssh_pkey=args.ssh_key,
            remote_bind_address=("127.0.0.1", args.remote_db_port),
            local_bind_address=("127.0.0.1", args.local_port),
            set_keepalive=30.0,
        )
        forwarder.start()

        forwarded_dsn = (
            f"{args.dsn} host=127.0.0.1 port={forwarder.local_bind_port} "
            "connect_timeout=10 sslmode=disable"
        )
    else:
        print("[System] Running in Local Mode (No SSH Tunnel).")
        forwarder = None
        # Use the provided DSN directly, but ensure it has a reasonable connect_timeout and sslmode disabled for local connections.
        # This allows the script to be flexible and run in environments where SSH tunneling is not necessary, such as when the database is running locally or on the same network.
        forwarded_dsn = f"{args.dsn} connect_timeout=10 sslmode=disable"
    
    # ======================================================================
    # [Dynamic Param] Auto-configure max_parallel_workers_per_gather
    # ======================================================================
    print(f"\n[System] Detecting environment limits from DB...")
    try:
        # Retrieve max_parallel_workers from DB
        temp_conn = psycopg2.connect(forwarded_dsn)
        with temp_conn.cursor() as cur:
            cur.execute("SHOW max_parallel_workers;")
            db_max_workers = int(cur.fetchone()[0])
        temp_conn.close()
        print(f"[System] Detected 'max_parallel_workers' = {db_max_workers}")
        
        # Generate dynamic levels
        target_steps = 6 
        if db_max_workers < target_steps:
            dynamic_levels = list(range(db_max_workers + 1))
        else:
            step_size = max(1, db_max_workers // (target_steps - 1))
            dynamic_levels = list(range(0, db_max_workers, step_size))
            if dynamic_levels[-1] != db_max_workers:
                dynamic_levels.append(db_max_workers)
            dynamic_levels = sorted(list(set(dynamic_levels)))

        print(f"[System] Configured 'max_parallel_workers_per_gather' levels: {dynamic_levels}")

        P1_PARAM_SPECS["max_parallel_workers_per_gather"] = {
            "min": 0,
            "max": len(dynamic_levels) - 1,
            "fmt": lambda x, levels=dynamic_levels: f"{levels[int(round(x))]}",
            "cast": int,
            "levels": dynamic_levels,
            "levels_unit": "native",
        }
        print("[System] Injected dynamic levels into 'max_parallel_workers_per_gather'.")

    except Exception as e:
        print(f"[Warning] Failed to auto-detect limits: {e}")

    # ======================================================================
    
    sanitized_query_name = args.queries.replace(" ", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"training_log/{sanitized_query_name}/{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_log = report_dir / f"{sanitized_query_name}_experiment_report.log"
    
    header_info = [
        "="*60,
        f"Experiment Report: {sanitized_query_name}",
        f"Timestamp: {timestamp}",
        f"Params: Total P1={args.total_p1}, Total P2={args.total_p2}, Total P3={args.total_p3}",
        "="*60,
        ""
    ]
    append_to_master_log(report_log, header_info)
    
    try:
        p1_converged_config = {}
        p2_converged_config = {}
        p3_converged_config = {}
        p1_best_latency = None
        p2_best_latency = None
        p3_best_latency = None

        # ======================================================================
        # Phase 1: Planner (Session)
        # ======================================================================
        if args.total_p1 > 0:
            print("\n" + "="*60)
            print(f"[Phase 1] Tuning Planner parameters (Session Mode) - {args.total_p1} steps")
            print("="*60)
            
            env_p1 = PgConfEnv(
                dsn=forwarded_dsn,
                ssh_client=None,
                ssh_password=args.ssh_password,
                remote_conf_path=args.remote_conf,
                tuning_mode="session",
                tune_params=list(P1_PARAM_SPECS.keys()),
                param_specs=P1_PARAM_SPECS,
                workload=qs,
                schedule=args.schedule,
                episode_len=1,
                start_from_default=True,
                baseline_first_step=True,
                early_stop_factor=args.early_stop_factor,
                min_timeout_ms=args.min_timeout_ms,
                timeout_penalty=args.timeout_penalty
            )
            
            if args.total_p1 < 64:
                p1_n_steps = args.total_p1
                p1_batch_size = args.total_p1
                if p1_n_steps < 1: p1_n_steps = 1
                if p1_batch_size < 1: p1_batch_size = 1
            else:
                p1_n_steps = 64
                p1_batch_size = 64

            print(f"[Phase 1] PPO Init: n_steps={p1_n_steps}, batch_size={p1_batch_size}")
            
            model_p1 = PPO("MlpPolicy", env_p1, verbose=1, device="cpu", ent_coef=args.ent,
                           n_steps=p1_n_steps,
                           batch_size=p1_batch_size)
            
            p1_log_path = log_dir / f"{args.queries}_P1_steps{args.total_p1}.log"
            print(f"[Phase 1] Logging to {p1_log_path}")
            with open(p1_log_path, "w", encoding="utf-8") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                model_p1.learn(total_timesteps=args.total_p1, callback=[PPOLogger(), StepAnnealCB(args.total_p1)])

            print("[Phase 1] Analyzing Convergence...")
            p1_converged_config, p1_report_lines = extract_converged_params_from_log(p1_log_path, P1_PARAM_SPECS, last_n=20, threshold_ratio=DEFAULT_THRESHOLD)
            append_to_master_log(report_log, ["[Phase 1 Analysis Result]"] + p1_report_lines)
            
            # Retrieve the best latency from the environment after training, which will be used as baseline for P2's catastrophic stopping
            p1_best_latency = env_p1.best_latency_ms 
            p1_baseline_latency = env_p1.latency_baseline_ms
            print(f"[Phase 1] Baseline Latency: {p1_baseline_latency:.2f} ms")
            print(f"[Phase 1] Best Latency Achieved: {p1_best_latency:.2f} ms")
            
            if p1_baseline_latency > 0 and p1_best_latency:
                p1_improvement = (p1_baseline_latency - p1_best_latency) / p1_baseline_latency
                print(f"[Phase 1] Latency Improvement: {p1_improvement*100:.2f}%")
                
                # If the improvement is less than or equal to 2%, we consider it negligible and discard the converged parameters to prevent overfitting to GCM noise. 
                # This is a safeguard to ensure that we only apply changes that have a meaningful impact on performance.
                if p1_improvement <= 0.02: 
                    print("[Phase 1] [Warning] Improvement is negligible. Discarding converged parameters to prevent GCM noise.")
                    p1_converged_config = {}
                    append_to_master_log(report_log, [f"[Phase 1 Validation] Negligible improvement ({p1_improvement*100:.2f}% <= 2%). Parameters discarded."])

            print("[Phase 1] Persisting configs...")
            
            if args.ssh_host and args.ssh_host not in ["", "localhost", "127.0.0.1"]:
                print("[Phase 1] Connecting Temporary SSH...")
                temp_ssh = create_ssh_client(args)
            else:
                print("[Phase 1] Using Local Command (No SSH)...")
                temp_ssh = None
                
            try:
                env_p1.ssh_client = temp_ssh
                
                if p1_converged_config:
                    print("[Phase 1] Applying Converged Config to DB...")
                    env_p1._update_remote_config_and_restart(p1_converged_config)
                    print("[Phase 1] DB Restarted. Environment is ready for Phase 2.")
                else:
                    print("[Warning] No parameters converged in Phase 1! Reverting to full defaults.")
                    env_p1._update_remote_config_and_restart({})
            except Exception as e:
                print(f"[Error] Failed to apply P1 config: {e}")
            finally:
                if temp_ssh:
                    temp_ssh.close()
                    print("[Phase 1] Temporary SSH connection closed.")
            
            env_p1.close()
        
        p1_p2_merged_params = p1_converged_config.copy()

        # ======================================================================
        # Phase 2: JIT & GEQO (Session)
        # ======================================================================
        if args.total_p2 > 0:
            print("\n" + "="*60)
            print(f"[Phase 2] Tuning JIT & GEQO parameters (Session Mode) - {args.total_p2} steps")
            print("="*60)
            
            env_p2 = PgConfEnv(
                dsn=forwarded_dsn,
                ssh_client=None,
                ssh_password=args.ssh_password,
                remote_conf_path=args.remote_conf,
                tuning_mode="session",
                tune_params=list(P2_PARAM_SPECS.keys()),
                param_specs=P2_PARAM_SPECS,
                workload=qs,
                schedule=args.schedule,
                episode_len=1,
                start_from_default=False,
                baseline_first_step=True,
                early_stop_factor=args.early_stop_factor,
                min_timeout_ms=args.min_timeout_ms,
                timeout_penalty=args.timeout_penalty,
                fixed_params={},
                initial_baseline_ms=p1_best_latency
            )
            
            if args.total_p2 < 64:
                p2_n_steps = args.total_p2
                p2_batch_size = args.total_p2
                if p2_n_steps < 1: p2_n_steps = 1
                if p2_batch_size < 1: p2_batch_size = 1
            else:
                p2_n_steps = 64
                p2_batch_size = 64

            print(f"[Phase 2] PPO Init: n_steps={p2_n_steps}, batch_size={p2_batch_size}")
            
            model_p2 = PPO("MlpPolicy", env_p2, verbose=1, device="cpu", ent_coef=args.ent,
                           n_steps=p2_n_steps,
                           batch_size=p2_batch_size)
            
            p2_log_path = log_dir / f"{args.queries}_P2_steps{args.total_p2}.log"
            print(f"[Phase 2] Logging to {p2_log_path}")
            
            p2_callbacks = [
                PPOLogger(), 
                StepAnnealCB(args.total_p2),
                ConvergenceStoppingCallback(patience=640, min_delta_ratio=0.01, catastrophic_patience=20)
            ]
            
            with open(p2_log_path, "w", encoding="utf-8") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                model_p2.learn(total_timesteps=args.total_p2, callback=p2_callbacks)
            
            print("[Phase 2] Analyzing Convergence...")
            p2_converged_config, p2_report_lines = extract_converged_params_from_log(p2_log_path, P2_PARAM_SPECS, last_n=20, threshold_ratio=DEFAULT_THRESHOLD)
            append_to_master_log(report_log, ["[Phase 2 Analysis Result]"] + p2_report_lines)
            
            # Retrieve the best latency from the environment after training, which will be used as baseline for P2's catastrophic stopping
            p2_best_latency = env_p2.best_latency_ms 
            p2_baseline_latency = env_p2.latency_baseline_ms
            print(f"[Phase 2] Baseline Latency (from P1): {p2_baseline_latency:.2f} ms")
            print(f"[Phase 2] Best Latency Achieved: {p2_best_latency:.2f} ms")
            
            if p2_baseline_latency > 0 and p2_best_latency:
                p2_improvement = (p2_baseline_latency - p2_best_latency) / p2_baseline_latency
                print(f"[Phase 2] Latency Improvement: {p2_improvement*100:.2f}%")
                
                if p2_improvement <= 0.02:
                    print("[Phase 2] [Warning] Improvement is negligible. Discarding converged parameters to prevent GCM noise.")
                    p2_converged_config = {}
                    append_to_master_log(report_log, [f"[Phase 2 Validation] Negligible improvement ({p2_improvement*100:.2f}% <= 2%). Parameters discarded."])
            
            if p2_converged_config:
                print(f"[Phase 2] Identified {len(p2_converged_config)} converged params: {p2_converged_config}")
                
                # Merge P1 & P2 converged params only
                # There is no need to apply to DB and restart here, as P3 will handle it
                p1_p2_merged_params.update(p2_converged_config) 
            else:
                # If no params converged, keep P1 params only
                print("[Warning] No parameters converged in Phase 2! P2 params remain defaults.")
            
            env_p2.close()
            
        # ======================================================================
        # Phase 3: Memory & I/O Resources (Restart)
        # ======================================================================
        if args.total_p3 > 0:
            print("\n" + "="*60)
            print(f"[Phase 3] Tuning Memory & I/O resources parameters (Restart Mode) - {args.total_p3} steps")
            print("="*60)
            
            if args.ssh_host and args.ssh_host not in ["", "localhost", "127.0.0.1"]:
                # Create persistent SSH connection for P3 cause it needs to restart DB multiple times
                print("[System] Establishing persistent SSH connection for Phase 3...")
                ssh_ctrl = create_ssh_client(args) 
                
                # Before starting Phase 3, ensure that the remote PostgreSQL is active and can be restarted via SSH commands.
                print("[System] Verifying remote PostgreSQL status before Phase 3...")
                db_ready = False
                for attempt in range(10): 
                    try:
                        stdin, stdout, stderr = ssh_ctrl.exec_command("systemctl is-active postgresql")
                        stdin.write(args.ssh_password + '\n')
                        stdin.flush()
                        status = stdout.read().decode().strip()
                        
                        if status == "active":
                            print("[System] Remote PostgreSQL is ACTIVE and READY.")
                            db_ready = True
                            break
                        else:
                            print(f"[System] Remote DB status is '{status}'. Attempting to start/restart (Attempt {attempt+1}/10)...")
                            stdin, stdout, stderr = ssh_ctrl.exec_command("sudo -S systemctl restart postgresql")
                            stdin.write(args.ssh_password + '\n')
                            stdin.flush()
                            time.sleep(5)
                    except Exception as e:
                        print(f"[Warning] Health check failed: {e}")
                        time.sleep(2)
                
                if not db_ready:
                    print("[Critical] Could not start PostgreSQL after multiple attempts. Phase 3 might fail.")
                
                print(f"[Phase 3] Context: Inherited {len(p1_p2_merged_params)} fixed params from P1 & P2")
                
            else:
                print("[System] Running Phase 3 in Local Mode (No SSH). Skipping SSH check.")
                ssh_ctrl = None
                
            phase3_initial_baseline = p2_best_latency if p2_best_latency is not None else p1_best_latency

            env_p3 = PgConfEnv(
                dsn=forwarded_dsn,
                ssh_client=ssh_ctrl,
                ssh_password=args.ssh_password,
                remote_conf_path=args.remote_conf,
                tuning_mode="restart",
                tune_params=list(P3_PARAM_SPECS.keys()),
                param_specs=P3_PARAM_SPECS,
                workload=qs,
                schedule=args.schedule,
                episode_len=1,
                start_from_default=False,
                baseline_first_step=True,
                early_stop_factor=args.early_stop_factor,
                min_timeout_ms=args.min_timeout_ms,
                timeout_penalty=args.timeout_penalty,
                fixed_params=p1_p2_merged_params,
                initial_baseline_ms=phase3_initial_baseline
            )
            
            if args.total_p3 < 64:
                p3_n_steps = args.total_p3
                p3_batch_size = args.total_p3
                if p3_n_steps < 1: p3_n_steps = 1
                if p3_batch_size < 1: p3_batch_size = 1
            else:
                p3_n_steps = 64
                p3_batch_size = 64

            print(f"[Phase 3] PPO Init: n_steps={p3_n_steps}, batch_size={p3_batch_size}")
            
            model_p3 = PPO("MlpPolicy", env_p3, verbose=1, device="cpu", ent_coef=args.ent,
                           n_steps=p3_n_steps,
                           batch_size=p3_batch_size)
            
            p3_log_path = log_dir / f"{args.queries}_P3_steps{args.total_p3}.log"
            print(f"[Phase 3] Logging to {p3_log_path}")
            
            p3_callbacks = [
                PPOLogger(), 
                StepAnnealCB(args.total_p3),
                ConvergenceStoppingCallback(patience=320, min_delta_ratio=0.01, catastrophic_patience=20) 
            ]
            
            with open(p3_log_path, "w", encoding="utf-8") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                model_p3.learn(total_timesteps=args.total_p3, callback=p3_callbacks)

            print("[Phase 3] Analyzing Convergence...")
            p3_converged_config, p3_report_lines = extract_converged_params_from_log(p3_log_path, P3_PARAM_SPECS, last_n=20, threshold_ratio=DEFAULT_THRESHOLD)
            append_to_master_log(report_log, ["[Phase 3 Analysis Result]"] + p3_report_lines)
            
            p3_best_latency = env_p3.best_latency_ms
            p3_baseline_latency = env_p3.latency_baseline_ms
            print(f"[Phase 3] Baseline Latency (from previous phase/current baseline): {p3_baseline_latency:.2f} ms")
            print(f"[Phase 3] Best Latency Achieved: {p3_best_latency:.2f} ms")
            
            if p3_baseline_latency > 0 and p3_best_latency:
                p3_improvement = (p3_baseline_latency - p3_best_latency) / p3_baseline_latency
                print(f"[Phase 3] Latency Improvement: {p3_improvement*100:.2f}%")
                
                if p3_improvement <= 0.02:
                    print("[Phase 3] [Warning] Improvement is negligible. Discarding converged parameters to prevent GCM noise.")
                    p3_converged_config = {}
                    append_to_master_log(report_log, [f"[Phase 3 Validation] Negligible improvement ({p3_improvement*100:.2f}% <= 2%). Parameters discarded."])
            
            print("[System] Training finished. Resetting tuning config to DEFAULTS for cleanup...")
            try:
                env_p3.fixed_params = {}
                env_p3._update_remote_config_and_restart({}) 
            except Exception as e:
                print(f"[Warning] Cleanup restart failed: {e}")

            env_p3.close()
            if ssh_ctrl:
                ssh_ctrl.close()
                ssh_ctrl = None
        
        # ======================================================================
        # Final Summary
        # ======================================================================
        print(f"\n[System] Generating Summary to {report_log} ...")

        summary_lines = []
        summary_lines.append("="*60)
        summary_lines.append("[Final Consolidated Summary]")
        summary_lines.append(f"{'Parameter':<30} | {'Phase':<5} | {'Status':<10} | {'Converged Value'}")
        summary_lines.append("-" * 75)
        
        # Helper to format lines
        def add_rows(specs, data, phase_label):
            for param in specs.keys():
                spec = specs[param]
                display_name = param
                if spec.get("is_virtual", False) and "map_to" in spec:
                    display_name = spec["map_to"][0]

                if param in data:
                    status = "STABLE"
                    val = data[param]
                else:
                    is_active = (
                        (phase_label == "P1" and args.total_p1 > 0) or 
                        (phase_label == "P2" and args.total_p2 > 0) or 
                        (phase_label == "P3" and args.total_p3 > 0)
                    )
                    status = "UNSTABLE" if is_active else "SKIP"
                    val = "N/A"
                
                summary_lines.append(f"{display_name:<30} | {phase_label:<5} | {status:<10} | {val}")

        add_rows(P1_PARAM_SPECS, p1_converged_config, "P1")
        add_rows(P2_PARAM_SPECS, p2_converged_config, "P2")
        add_rows(P3_PARAM_SPECS, p3_converged_config, "P3")
        
        summary_lines.append("="*60)
        append_to_master_log(report_log, summary_lines)
        print("\n".join(summary_lines))
        print(f"[System] All logs saved to: {log_dir}")

    finally:
        if ssh_ctrl: ssh_ctrl.close()
        
        if forwarder and forwarder.is_active:
            forwarder.stop()

if __name__ == "__main__":
    main()