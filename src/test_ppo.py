# test_ppo_v11.py
# cosmos demo machine
# 3-Phase Hierarchical Training
# Phase 1: Planner (Session)
# Phase 2: JIT & GEQO (Session) 
# Phase 3: Memory & I/O Resources (Restart)
# Added Convergence Stopping Mechanism for P2/P3 phases
# Added Dynamic Parallel Degree Parameter Mapping
# Fixed extract_converged_params_from_log() to excluded cache-influenced steps
# Added ssh_host check for local mode

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

from tpch_queryspecs import Q_MAP
from param_specs import P1_PARAM_SPECS, P2_PARAM_SPECS, P3_PARAM_SPECS

DEFAULT_TOTAL_P1 = 2048
DEFAULT_TOTAL_P2 = 2048
DEFAULT_TOTAL_P3 = 2048
DEFAULT_THRESHOLD = 0.8
TEST_SQL = "Q1"

def resolve_virtual_params(config: dict, specs: dict) -> dict:
    """
    將收斂結果中的「虛擬參數」轉換回「真實 PostgreSQL 參數」。
    例如：將 {'parallel_degree': '20'} 轉換為 {'max_parallel_workers_per_gather': '20'}
    """
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
    
    # [修改後邏輯]
    # 1. 先排序 latency
    sorted_lats = sorted([d[0] for d in window_data])
    
    # 2. 取出第 20 百分位 (P20) 的數值作為基準 (Baseline)
    # 這能避開前 20% 極端快(Cache Hit) 的數據影響 Cutoff 計算
    p_index = int(len(sorted_lats) * 0.20) 
    baseline_lat = sorted_lats[p_index]
    
    # 3. 計算 Cutoff (依然使用原本的 tolerance)
    cutoff_lat = baseline_lat * (1.0 + latency_tolerance)
    
    # 4. 過濾 (注意：這裡依然會保留那些比 baseline 更快的 Cache Hit 數據，因為它們也是有效的執行計畫)
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
        
        # [新增] 顯示名稱轉換邏輯
        spec = param_specs[param]
        display_name = param
        if spec.get("is_virtual", False) and "map_to" in spec:
            # 如果是虛擬參數，顯示它對應的第一個真實參數名稱
            display_name = spec["map_to"][0]

        # [修改] 使用 display_name 來列印 log
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
    """建立並回傳一個新的 SSH Client"""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        args.ssh_host, port=args.ssh_port, username=args.ssh_user, 
        password=args.ssh_password, key_filename=args.ssh_key
    )
    # 設定 Keepalive 防止短時間操作內斷線
    client.get_transport().set_keepalive(60)
    return client

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
    ap.add_argument("--dsn", default="dbname=tpch10 user=wettin password=Qwer1234!", help="Base DSN without host/port; we will inject host=127.0.0.1 and forwarded port.")

    ap.add_argument("--ssh-host", default="")
    ap.add_argument("--ssh-port", type=int, default=22)
    ap.add_argument("--ssh-user", default="")
    ap.add_argument("--ssh-key", default=None)
    ap.add_argument("--ssh-password", default="")
    ap.add_argument("--local-port", type=int, default=5432)
    ap.add_argument("--remote-conf", default="/var/lib/postgresql/data/auto_tuning.conf")

    ap.add_argument("--queries", default=TEST_SQL)
    ap.add_argument("--schedule", default="single", choices=["single","round_robin","random"])
    
    ap.add_argument("--total-p1", type=int, default=DEFAULT_TOTAL_P1)
    ap.add_argument("--total-p2", type=int, default=DEFAULT_TOTAL_P2)
    ap.add_argument("--total-p3", type=int, default=DEFAULT_TOTAL_P3)
    
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--early-stop-factor", type=float, default=5.0)
    ap.add_argument("--min-timeout-ms", type=int, default=3000)
    ap.add_argument("--timeout-penalty", type=float, default=-100.0)
    
    args = ap.parse_args()
    
    ssh_ctrl = None
    forwarder = None
    
    qs = [Q_MAP[name.strip()] for name in args.queries.split(',') if name.strip()]

    # 1. 啟動 SSH Tunnel (必須常駐，因為 P1/P2/P3 訓練都需要它來傳送 SQL)
    if args.ssh_host and args.ssh_host not in ["", "localhost", "127.0.0.1"]:
        print(f"[System] Starting SSH Tunnel to {args.ssh_host}...")
        forwarder = SSHTunnelForwarder(
            (args.ssh_host, args.ssh_port),
            ssh_username=args.ssh_user,
            ssh_password=args.ssh_password,
            ssh_pkey=args.ssh_key,
            remote_bind_address=("127.0.0.1", 5432),
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
        # 直接使用傳入的 DSN，或者預設連線到 Docker 內部的 DB
        # 如果是 Docker 內部互連，通常 host=localhost 即可 (因為 Python 和 DB 在同一個容器)
        forwarded_dsn = f"{args.dsn} connect_timeout=10 sslmode=disable"
    
    # ======================================================================
    # [Dynamic Param] Auto-configure parallel_degree
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
        
        # Generate dynamic levels for parallel_degree
        target_steps = 6 
        if db_max_workers < target_steps:
            dynamic_levels = list(range(db_max_workers + 1))
        else:
            step_size = max(1, db_max_workers // (target_steps - 1))
            dynamic_levels = list(range(0, db_max_workers, step_size))
            if dynamic_levels[-1] != db_max_workers:
                dynamic_levels.append(db_max_workers)
            dynamic_levels = sorted(list(set(dynamic_levels)))

        print(f"[System] Configured 'parallel_degree' levels: {dynamic_levels}")

        if "max_parallel_workers_per_gather" in P1_PARAM_SPECS:
            del P1_PARAM_SPECS["max_parallel_workers_per_gather"]
            print("[System] Removed raw 'max_parallel_workers_per_gather' from action space.")

        # Inject dynamic virtual parameter spec, mapping to max_parallel_workers_per_gather
        P1_PARAM_SPECS["parallel_degree"] = {
            "is_virtual": True,
            "map_to": ["max_parallel_workers_per_gather"],
            "min": 0,
            "max": len(dynamic_levels) - 1, 
            "fmt": lambda x, levels=dynamic_levels: levels[int(round(x))],
            "cast": int,
        }
        print("[System] Injected dynamic 'parallel_degree' into P1_PARAM_SPECS.")

    except Exception as e:
        print(f"[Warning] Failed to auto-detect limits: {e}")

    # ======================================================================
    
    sanitized_query_name = args.queries.replace(" ", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"training_log/{sanitized_query_name}/{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)
    report_log = log_dir / f"{sanitized_query_name}_experiment_report.log"
    
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
            
            p1_callbacks = [
                PPOLogger(), 
                StepAnnealCB(args.total_p1),
                ConvergenceStoppingCallback(patience=1024, min_delta_ratio=0.01) 
            ]
            
            with open(p1_log_path, "w", encoding="utf-8") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                model_p1.learn(total_timesteps=args.total_p1, callback=p1_callbacks)

            print("[Phase 1] Analyzing Convergence...")
            p1_converged_config, p1_report_lines = extract_converged_params_from_log(p1_log_path, P1_PARAM_SPECS, last_n=20, threshold_ratio=DEFAULT_THRESHOLD)
            append_to_master_log(report_log, ["[Phase 1 Analysis Result]"] + p1_report_lines)
            
            p1_real_config = resolve_virtual_params(p1_converged_config, P1_PARAM_SPECS)

            print("[Phase 1] Persisting configs...")
            
            # Check if we need SSH, in Docker environment we can skip SSH and apply config directly
            if args.ssh_host and args.ssh_host not in ["", "localhost", "127.0.0.1"]:
                print("[Phase 1] Connecting Temporary SSH...")
                temp_ssh = create_ssh_client(args)
            else:
                print("[Phase 1] Using Local Command (No SSH)...")
                temp_ssh = None
                
            try:
                env_p1.ssh_client = temp_ssh
                
                if p1_converged_config:
                    print(f"[Phase 1] Identified converged params (Virtual): {p1_converged_config}")
                    print(f"[Phase 1] Resolving to Real Params: {p1_real_config}")
                    print("[Phase 1] Applying Converged Config to DB...")
                    env_p1._update_remote_config_and_restart(p1_real_config)
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
        
        p1_p2_merged_params = p1_real_config.copy()

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
                fixed_params={}
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
                ConvergenceStoppingCallback(patience=256, min_delta_ratio=0.01) 
            ]
            
            with open(p2_log_path, "w", encoding="utf-8") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                model_p2.learn(total_timesteps=args.total_p2, callback=p2_callbacks)
            
            print("[Phase 2] Analyzing Convergence...")
            p2_converged_config, p2_report_lines = extract_converged_params_from_log(p2_log_path, P2_PARAM_SPECS, last_n=20, threshold_ratio=DEFAULT_THRESHOLD)
            append_to_master_log(report_log, ["[Phase 2 Analysis Result]"] + p2_report_lines)
            
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
                
                # [Health Check] 確保 P3 開始前 DB 是活著的
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
                fixed_params=p1_p2_merged_params
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
                ConvergenceStoppingCallback(patience=128, min_delta_ratio=0.01) 
            ]
            
            with open(p3_log_path, "w", encoding="utf-8") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                model_p3.learn(total_timesteps=args.total_p3, callback=p3_callbacks)

            print("[Phase 3] Analyzing Convergence...")
            p3_converged_config, p3_report_lines = extract_converged_params_from_log(p3_log_path, P3_PARAM_SPECS, last_n=20, threshold_ratio=DEFAULT_THRESHOLD)
            append_to_master_log(report_log, ["[Phase 3 Analysis Result]"] + p3_report_lines)
            
            print("[System] Training finished. Resetting remote config to DEFAULTS for cleanup...")
            try:
                env_p3.fixed_params = {}
                env_p3._update_remote_config_and_restart({}) 
            except Exception as e:
                print(f"[Warning] Cleanup restart failed: {e}")

            env_p3.close()
            if ssh_ctrl:
                ssh_ctrl.close()
        
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
                # [新增] 檢查是否為虛擬參數，如果是，取出它對應的真實名稱
                spec = specs[param]
                display_name = param
                if spec.get("is_virtual", False) and "map_to" in spec:
                    # 取出 map_to 列表中的第一個真實參數名稱 (例如 max_parallel_workers_per_gather)
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
                
                # [修改] 這裡使用 display_name 來列印
                summary_lines.append(f"{display_name:<30} | {phase_label:<5} | {status:<10} | {val}")

        add_rows(P1_PARAM_SPECS, p1_converged_config, "P1")
        add_rows(P2_PARAM_SPECS, p2_converged_config, "P2")
        add_rows(P3_PARAM_SPECS, p3_converged_config, "P3")
        
        summary_lines.append("="*60)
        append_to_master_log(report_log, summary_lines)
        print("\n".join(summary_lines))
        print(f"[System] All logs saved to: {log_dir}")

    finally:
        # 確保例外發生時，如果是 P3 階段建立的 ssh_ctrl 能被關閉
        if ssh_ctrl: ssh_ctrl.close()
        
        if args.ssh_host and args.ssh_host not in ["", "localhost", "127.0.0.1"]:
            forwarder.stop()

if __name__ == "__main__":
    main()