# merge_configs.py

import argparse
import psycopg2
import time
import re
import pprint
import sys
import os
import csv
import subprocess
import glob
import shlex
import paramiko
from copy import deepcopy
from pathlib import Path
from sshtunnel import SSHTunnelForwarder
from typing import Dict, Set, Tuple, Any, List
from datetime import datetime

# The frequency of each query in the workload, defaulting to 1 if not specified. This is used in the regression-aware scoring.
QUERY_FREQUENCIES = {

}

PHASE_ORDER = [
    # --- Phase 1: Core Planner & Parallelism ---
    "work_mem", "hash_mem_multiplier", 
    "parallel_leader_participation", "max_parallel_workers_per_gather",
    "effective_cache_size", "cpu_tuple_cost", "cpu_index_tuple_cost", 
    "cpu_operator_cost", "seq_page_cost", "random_page_cost", 
    "min_parallel_table_scan_size", "min_parallel_index_scan_size", 
    "parallel_tuple_cost", "parallel_setup_cost", "enable_seqscan", 
    "enable_indexscan", "enable_indexonlyscan", "enable_bitmapscan", 
    "enable_sort", "enable_incremental_sort", "enable_hashagg", 
    "enable_material", "enable_memoize", "enable_nestloop", 
    "enable_mergejoin", "enable_hashjoin", "enable_gathermerge", 
    "enable_parallel_hash",
    
    # --- Phase 2: Advanced Optimizer (JIT, GEQO) ---
    "jit_above_cost", "jit_optimize_above_cost", "jit_inline_above_cost",
    "geqo", "geqo_threshold", "geqo_effort", "geqo_pool_size",
    "geqo_generations", "geqo_selection_bias", "jit",
    "from_collapse_limit", "join_collapse_limit",
    
    # --- Phase 3: System Resources & Vacuum ---
    "min_dynamic_shared_memory", "shared_buffers", "maintenance_work_mem",
    "autovacuum_work_mem", "effective_io_concurrency", "maintenance_io_concurrency"
]

def get_param_priority(param_name: str) -> int:
    try:
        return PHASE_ORDER.index(param_name)
    except ValueError:
        return 999

try:
    from tpch_queryspecs import SQL_MAP
    from param_specs import PARAM_SPECS
except ImportError:
    print("Error: Failed to import 'tpch_queryspecs.py' or 'param_specs.py'.", file=sys.stderr)
    sys.exit(1)


class ConfigMerger:
    def __init__(self, 
                 dsn: str, 
                 workload_q_map: Dict[str, Any], 
                 param_specs: Dict[str, Dict],
                 all_tuned_params: List[str],
                 ssh_client: paramiko.SSHClient | None = None,
                 ssh_password: str | None = None,
                 report_dir: str = "./training_log/experiment_reports",
                 remote_conf_path: str | None = None,
                 degradation_lambda: float = 2.5,
                 degradation_power: float = 2.0,
                 ):
        
        self.dsn = dsn
        self.workload_q_map = workload_q_map
        self.param_specs = param_specs
        self.all_tuned_params = all_tuned_params
        
        self._system_baseline = None
        self._convergence_data: Dict[str, Dict[str, Set[str]]] | None = None
        
        self.EVAL_TIMEOUT_MS = 30000
        self.scoring_profile = "regression_aware"

        if degradation_lambda < 0:
            raise ValueError("degradation_lambda must be greater than or equal to 0")
        if degradation_power <= 0:
            raise ValueError("degradation_power must be greater than 0")
        self.DEGRADATION_LAMBDA = degradation_lambda
        self.DEGRADATION_POWER = degradation_power
        self.timeout_blacklist: Set[Tuple[str, str]] = set()
        
        self._param_regexes = {
            p: re.compile(rf"\b{p}=([^,\s]+)") 
            for p in self.all_tuned_params
        }
        
        self.log_data: List[Dict[str, Any]] = []
        self.ssh_client = ssh_client
        self.ssh_password = ssh_password
        
        self.report_dir = Path(report_dir)
        default_conf_path = os.path.join(os.getenv("PGDATA", "/var/lib/pgsql/data"), "auto_tuning.conf")
        self.remote_conf_path = remote_conf_path or os.getenv("REMOTE_CONF_PATH", default_conf_path)

        print(f"ConfigMerger initialized. Scoring profile: {self.scoring_profile}")
        print(f"PostgreSQL tuning config file: {self.remote_conf_path}")

    # --- PostgreSQL Configuration Management (Local or SSH) ---

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

    def _write_local_config(self, config_content: str) -> bool:
        try:
            os.makedirs(os.path.dirname(self.remote_conf_path), exist_ok=True)
            with open(self.remote_conf_path, "w", encoding="utf-8") as f:
                f.write(config_content)
            subprocess.run(["chown", "postgres:postgres", self.remote_conf_path], check=True)
            subprocess.run(["chmod", "644", self.remote_conf_path], check=True)
            return True
        except Exception as e:
            print(f"     -> Error: Failed to update local configuration file: {e}")
            return False

    def _restart_remote_postgresql(self) -> bool:
        """Restart PostgreSQL in local Docker mode or on the remote SSH host."""
        if self.ssh_client is None:
            try:
                pg_ctl = self._local_pg_ctl()
                pgdata = self._local_pgdata()
                command = f"{shlex.quote(pg_ctl)} -D {shlex.quote(pgdata)} -w restart"
                print("     -> [System] Restarting local PostgreSQL...")
                subprocess.run(["su", "-", "postgres", "-c", command], check=True)
                time.sleep(2.0)
                return True
            except Exception as e:
                print(f"     -> Error: Failed to restart local PostgreSQL: {e}")
                return False

        sudo_flag = "-S" if self.ssh_password else "-n"
        cmd = f"sudo {sudo_flag} systemctl restart postgresql"
        print("     -> [System] Restarting remote PostgreSQL...")
        try:
            stdin, stdout, stderr = self.ssh_client.exec_command(cmd, get_pty=True)
            if self.ssh_password:
                stdin.write(self.ssh_password + "\n")
                stdin.flush()
            exit_status = stdout.channel.recv_exit_status()
            if exit_status != 0:
                print(f"     -> Error: Failed to restart PostgreSQL: {stderr.read().decode()}")
                return False
            time.sleep(8.0)
            return True
        except Exception as e:
            print(f"     -> Error: Exception occurred while restarting PostgreSQL: {e}")
            return False

    def _update_remote_config(self, params: Dict[str, str]) -> bool:
        """Update auto_tuning.conf locally or through SSH/SFTP."""
        config_lines = []
        for key, value in params.items():
            p_spec = self.param_specs.get(key, {})
            if p_spec.get("is_virtual"):
                for real_key in p_spec.get("map_to", []):
                    config_lines.append(f"{real_key} = '{value}'")
            else:
                config_lines.append(f"{key} = '{value}'")
        config_content = "\n".join(config_lines)

        if self.ssh_client is None:
            return self._write_local_config(config_content)

        temp_path = "/tmp/auto_tuning.tmp"
        target_path = self.remote_conf_path
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
                print(f"     -> Error: Failed to update configuration file: {stderr.read().decode()}")
                return False
            return True
        except Exception as e:
            print(f"     -> Error: Exception occurred while updating configuration file: {e}")
            return False

    # --- Helper Functions ---
    def _humanize_setting(self, setting: Any, unit: str | None) -> str:
        setting_str = str(setting).strip()
        
        if setting_str == '-1':
            return "-1"
            
        if not unit: 
            return setting_str
            
        try:
            if unit.endswith("kB"):
                mul = 1
                if unit != "kB":
                    prefix = unit[:-2]
                    if prefix.isdigit(): mul = int(prefix)
                kb = int(float(setting_str)) * mul
                mb = kb / 1024.0
                return f"{int(mb)}MB" if mb.is_integer() else f"{mb:.2f}MB"
            return f"{setting_str}{unit}"
        except Exception:
            return setting_str

    def parse_logs(self, query_names: List[str]):
        print(f"\n--- Step 1: Parsing Training Summary Reports ---")
        
        reverse_param_map = {}
        for p_name, p_info in self.param_specs.items():
            if p_info.get("is_virtual"):
                for real_p in p_info.get("map_to", []):
                    reverse_param_map[real_p] = p_name
            else:
                reverse_param_map[p_name] = p_name

        convergence_data = {q: {p: set() for p in self.all_tuned_params} for q in query_names}
        
        for q_name in query_names:
            log_file = self.report_dir / f"{q_name}_experiment_report.log"
            if not log_file.exists():
                print(f"Warning: Report not found {log_file}, skipping {q_name}.")
                continue

            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            in_summary_block = False
            stable_count = 0
            
            for line in lines:
                line = line.strip()
                if line == "[Final Consolidated Summary]":
                    in_summary_block = True
                    continue
                if in_summary_block and line.startswith("=="):
                    break
                    
                if in_summary_block:
                    if line.startswith("Parameter") or line.startswith("---"):
                        continue
                    
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) >= 4:
                        real_param_name = parts[0]
                        status = parts[2]
                        converged_value = parts[3]
                        
                        gcm_param_name = reverse_param_map.get(real_param_name, real_param_name)
                        
                        if gcm_param_name in self.all_tuned_params:
                            if status == "STABLE" and converged_value != "N/A":
                                convergence_data[q_name][gcm_param_name].add(converged_value)
                                stable_count += 1
                                
            print(f"  {q_name}: Successfully parsed, found {stable_count} STABLE parameters")
            
        self._convergence_data = convergence_data
        print("Training summary reports parsing completed.")

    def classify_params(self) -> Tuple[Dict[str, str], Set[str], Set[str]]:
        if self._convergence_data is None: raise RuntimeError("Need to run parse_logs first")

        print("\n--- Step 2: Classifying Parameters ---")
        P_non_conflicting = {}
        P_conflict = set()
        
        for q_name in self.workload_q_map.keys():
            for param in self.all_tuned_params:
                if param in P_conflict: continue 

                observed_values = self._convergence_data[q_name].get(param)
                if not observed_values or len(observed_values) != 1: continue
                
                converged_value = next(iter(observed_values))
                if param not in P_non_conflicting:
                    P_non_conflicting[param] = converged_value
                elif P_non_conflicting[param] != converged_value:
                    del P_non_conflicting[param]
                    P_conflict.add(param)

        P_unconverged = set(self.all_tuned_params) - P_non_conflicting.keys() - P_conflict
        return P_non_conflicting, P_conflict, P_unconverged
    
    def get_system_baseline_config(self) -> Dict[str, str]:
        if self._system_baseline:
            return self._system_baseline
        
        print("\n--- Getting System Baseline Configuration ---")
        
        # 1. Clear auto_tuning.conf and restart DB to ensure we read the underlying configuration (User Config)
        print("  >> Clearing auto_tuning.conf and restarting DB to ensure reading the underlying configuration...")
        if not self._update_remote_config({}):
            raise RuntimeError("Failed to clear configuration file, cannot obtain Baseline.")
        if not self._restart_remote_postgresql():
            raise RuntimeError("Failed to restart DB, cannot obtain Baseline.")

        defaults = {}
        conn = None
        try:
            # 2. Connect to PostgreSQL and query pg_settings for all tuned parameters
            # pg_settings will return the effective value
            for attempt in range(10):
                try:
                    conn = psycopg2.connect(self.dsn)
                    break
                except:
                    time.sleep(2)
            
            if not conn: raise RuntimeError("Failed to connect to database, cannot obtain Baseline")

            with conn.cursor() as cur:
                for p in self.all_tuned_params:
                    cur.execute("SELECT setting, unit FROM pg_settings WHERE name = %s", (p,))
                    row = cur.fetchone()
                    if row:
                        setting, unit = row
                        human_val = self._humanize_setting(setting, unit)
                        defaults[p] = human_val
                    else:
                        print(f"Warning: Parameter {p} not found in pg_settings, skipping.")

        except Exception as e:
            print(f"Error: Failed to obtain Baseline: {e}", file=sys.stderr)
            raise
        finally:
            if conn: conn.close()
                
        self._system_baseline = defaults
        print("Retrieved System Baseline (C_default). Sample parameters:")
        keys_to_show = ['shared_buffers', 'work_mem', 'max_connections', 'effective_cache_size']
        for k in keys_to_show:
            if k in defaults:
                print(f"  {k} = {defaults[k]}")
        
        return defaults

    
    def evaluate_workload(self, C: Dict[str, str], target_queries: List[str] | None = None, verbose_print: bool = False) -> Tuple[float, Dict[str, float]]:
        
        if target_queries is None:
            queries_to_run = list(self.workload_q_map.keys())
        else:
            queries_to_run = target_queries

        # 1. Update remote configuration and restart DB before evaluation
        if not self._update_remote_config(C):
            print("     -> Error: Failed to update configuration file.")
            return (float('inf'), {})

        # 2. Restart PostgreSQL to apply new configuration
        if not self._restart_remote_postgresql():
             print("     -> Error: Failed to restart DB.")
             return (float('inf'), {})
        
        total_latency = 0.0
        query_latencies = {}
        conn = None
        
        try:
            for attempt in range(10):
                try:
                    conn = psycopg2.connect(self.dsn)
                    conn.autocommit = True 
                    break
                except psycopg2.OperationalError:
                    time.sleep(2.0)
            
            if conn is None: raise RuntimeError("Failed to establish connection")

            with conn.cursor() as cur:
                cur.execute(f"SET statement_timeout = {self.EVAL_TIMEOUT_MS}")

                queries_to_run.sort(key=lambda x: int(x[1:]) if x[1:].isdigit() else 999)
                if verbose_print: print(f"     -> Executing Workload ({len(queries_to_run)} queries)...")

                for q_name in queries_to_run:
                    if q_name not in self.workload_q_map: continue
                    q_spec = self.workload_q_map[q_name]
                    t0 = time.perf_counter()
                    try:
                        cur.execute(q_spec.sql, q_spec.params)
                        cur.fetchall()
                        lat = (time.perf_counter() - t0) * 1000.0
                        total_latency += lat
                        query_latencies[q_name] = lat
                        if verbose_print: print(f"        {q_name}: {lat:.2f} ms")
                    except Exception as e:
                        print(f"        {q_name}: TIMEOUT/ERROR ({e})")
                        total_latency += self.EVAL_TIMEOUT_MS
                        query_latencies[q_name] = float(self.EVAL_TIMEOUT_MS)
        except Exception as e:
            print(f"Error: Evaluation failed: {e}")
            return (float('inf'), {})
        finally:
            if conn: conn.close()
        
        return (total_latency, query_latencies)

    def _get_conflict_options(self, P_conflict: Set[str]) -> Dict[str, List[str]]:
        options_set = {p: set() for p in P_conflict}
        for param in P_conflict:
            for q_name in self.workload_q_map.keys():
                if param in self._convergence_data[q_name]:
                    vals = self._convergence_data[q_name][param]
                    if len(vals) == 1:
                        options_set[param].add(next(iter(vals)))
        return {p: list(s) for p, s in options_set.items()}

    # --- Regression-Aware Scoring Helpers ---
    def _calculate_impact_score(self, q_name: str, 
                                current_latency: float, 
                                reference_latency: float,
                                max_latency: float, 
                                max_freq: float) -> Tuple[float, float, float, float, float, float]:
        """Return regression-aware impact and the degradation multiplier."""
        freq = QUERY_FREQUENCIES.get(q_name, 1)
        s_lat = current_latency / max_latency if max_latency > 0 else 0.0
        s_freq = freq / max_freq if max_freq > 0 else 0.0

        degradation = 0.0
        if reference_latency > 0:
            degradation = max(0.0, (current_latency - reference_latency) / reference_latency)

        degradation_term = degradation ** self.DEGRADATION_POWER
        multiplier = 1.0 + self.DEGRADATION_LAMBDA * degradation_term
        impact = (0.5 * s_lat + 0.5 * s_freq) * multiplier
        auxiliary = multiplier

        return impact, freq, s_lat, s_freq, degradation, auxiliary

    def _calculate_workload_score(self,
                                  query_latencies: Dict[str, float],
                                  reference_latencies: Dict[str, float],
                                  max_latency: float,
                                  max_freq: float,
                                  verbose_print: bool = False,
                                  label: str = "") -> Tuple[float, Dict[str, Dict[str, float]]]:
        """Return the regression-aware workload score; lower is better."""
        details: Dict[str, Dict[str, float]] = {}
        additive_score = 0.0

        if verbose_print:
            print(f"     [{self.scoring_profile} Score] {label}")

        for q_name, latency in query_latencies.items():
            reference_latency = reference_latencies.get(q_name, latency)
            impact, freq, s_lat, s_freq, degradation, auxiliary = self._calculate_impact_score(
                q_name,
                latency,
                reference_latency,
                max_latency,
                max_freq,
            )

            contribution = latency * impact
            additive_score += contribution

            details[q_name] = {
                "impact": impact,
                "degradation": degradation,
                "auxiliary": auxiliary,
                "contribution": contribution,
                "s_lat": s_lat,
                "s_freq": s_freq,
                "frequency": float(freq),
            }

            if verbose_print:
                extra = f"Multiplier={auxiliary:.4f}, Impact={impact:.4f}"
                print(
                    f"        {q_name}: Lat={latency:.2f}ms, "
                    f"Degradation={degradation*100:.2f}%, {extra}, "
                    f"Contribution={contribution:.4f}"
                )

        total_score = additive_score

        if verbose_print:
            print(f"        -> Total Regression-Aware Score: {total_score:.2f}")

        return total_score, details
    
    # --- Algorithm Main Body  ---
    def run_merge_algorithm(self) -> Dict[str, str]:
        # 1. Parse training summary reports to classify parameters into P_non_conflicting, P_conflict, P_unconverged
        self.parse_logs(list(self.workload_q_map.keys()))
        
        # 2. Classify parameters based on convergence data
        P_non_conflicting, P_conflict, P_unconverged = self.classify_params()
        
        # Get System Baseline Configuration
        C_default = self.get_system_baseline_config() 

        # The regression-aware score uses per-query system-default latency as the degradation reference.
        # This reference is intentionally separate from L_base.
        print(
            f"\n--- {self.scoring_profile}: Evaluating system-default query baselines ---"
        )
        L_default, q_lats_default = self.evaluate_workload(C_default, verbose_print=True)
        self.log_data.append({
            "Test_Type": "L_default", "Configuration": "C_default (System Default)",
            "Total_Latency_ms": L_default, **q_lats_default
        })
        if L_default == float('inf'):
            print("System-default evaluation failed; degradation weighting cannot be calculated.")
            return C_default
             
        # 3. Evaluate C_base (System Baseline + Non-Conflicting) to get L_base and per-query latencies
        print("\n--- Step 3: Evaluating C_base and L_base ---")
        C_base = dict(C_default)
        C_base.update(P_non_conflicting)
        
        print("C_base (System Baseline + Non-Conflicting) initialized.")
        print("\n Evaluating C_base to get L_base and per-query latencies...")
        L_base, q_lats_base = self.evaluate_workload(C_base, verbose_print=True)
        
        self.log_data.append({
            "Test_Type": "L_base", "Configuration": "C_base (System + Non-Conflict)",
            "Total_Latency_ms": L_base, **q_lats_base
        })
        
        if L_base == float('inf'):
            print("C_base evaluation failed (infinite latency), cannot proceed with merging. Returning C_base as final configuration.")
            return C_base

        conflict_options = self._get_conflict_options(P_conflict)
        P_conflict_remaining = set(P_conflict)
        if hasattr(self, 'timeout_blacklist'): self.timeout_blacklist.clear()
        
        # Pre-calculate global max latency and frequency for impact scoring
        latency_samples = list(q_lats_default.values()) + list(q_lats_base.values())
        global_max_latency = max(latency_samples) if latency_samples else 1.0
        all_freqs = [QUERY_FREQUENCIES.get(q, 1) for q in self.workload_q_map.keys()]
        global_max_freq = max(all_freqs) if all_freqs else 1.0

        regression_score_base, _ = self._calculate_workload_score(
            q_lats_base,
            q_lats_default,
            global_max_latency,
            global_max_freq,
            verbose_print=True,
            label="Initial C_base",
        )
        self.log_data[-1]["Regression_Aware_Score"] = regression_score_base

        score_description = (
            f"Impact=(0.50X+0.50Y)*(1+{self.DEGRADATION_LAMBDA:.2f}"
            f"*D^{self.DEGRADATION_POWER:g})"
        )
        print(f"\n[Tie-Breaker Info] Mode={self.scoring_profile}, {score_description}")

        # --- Step 4: Local Conflict Resolution (with Tie-Breaker) ---
        print(f"\n--- Step 4: Local Conflict Resolution (with Tie-Breaker) ---")
        P_resolved_locally = set()
        current_conflict_list = sorted(list(P_conflict_remaining), key=get_param_priority)

        for param in current_conflict_list:
            if param not in conflict_options: continue
            options = conflict_options[param]
            if len(options) != 2: continue 

            involved_queries = set()
            for val in options:
                for q, data in self._convergence_data.items():
                    if param in data and next(iter(data[param]), None) == val:
                        involved_queries.add(q)
            
            if not involved_queries: continue
            target_queries = list(involved_queries)
            evaluation_queries = list(self.workload_q_map.keys())
            
            current_val = C_base.get(param)
            candidate_val = next((v for v in options if v != current_val), None)
            if not candidate_val: continue

            print(f"\n  >> Conflict {param}: Current[{current_val}] vs Candidate[{candidate_val}]")
            print(
                f"     -> Evaluating the full workload ({len(evaluation_queries)} queries); "
                f"convergence voters: {', '.join(sorted(target_queries))}"
            )
            
            # Always evaluate the full workload. Restricting this measurement to
            # convergence voters can hide collateral regressions in queries that
            # did not converge for this parameter (for example, S-1 under
            # enable_nestloop).
            local_base, q_lats_local_base = self.evaluate_workload(
                C_base, evaluation_queries.copy(), verbose_print=True
            )
            C_test = C_base.copy()
            C_test[param] = candidate_val
            local_test, q_lats_local_test = self.evaluate_workload(
                C_test, evaluation_queries.copy(), verbose_print=True
            )
            
            if local_base == float('inf') or local_test == float('inf'): continue

            local_score_base, local_details_base = self._calculate_workload_score(
                q_lats_local_base,
                q_lats_default,
                global_max_latency,
                global_max_freq,
                verbose_print=True,
                label=f"Current {param}={current_val}",
            )
            local_score_test, local_details_test = self._calculate_workload_score(
                q_lats_local_test,
                q_lats_default,
                global_max_latency,
                global_max_freq,
                verbose_print=True,
                label=f"Candidate {param}={candidate_val}",
            )

            raw_gain = local_base - local_test
            weighted_gain = local_score_base - local_score_test
            threshold = local_score_base * 0.02 # Preserve the original 2% dead zone.
            print(
                f"     Raw Base: {local_base:.2f}ms, Raw Test: {local_test:.2f}ms "
                f"| Raw Gain: {raw_gain:.2f}ms"
            )
            print(
                f"     Weighted Base: {local_score_base:.2f}, Weighted Test: {local_score_test:.2f} "
                f"| Weighted Gain: {weighted_gain:.2f} (2% Thres: {threshold:.2f})"
            )
            
            should_switch = False
            
            # --- Tie-Breaker ---
            if weighted_gain > threshold:
                should_switch = True
                print(f"     [Result] Regression-aware score improved by > 2% -> Switch")
            elif weighted_gain < -threshold:
                should_switch = False
                print(f"     [Result] Regression-aware score worsened by > 2% -> Keep")
            else:
                print(f"     [Result] Diff Small -> Activate Tie-Breaker")
                
                score_current_total = 0.0
                score_candidate_total = 0.0
                
                for q in target_queries:
                    favored_val = next(iter(self._convergence_data[q].get(param, [])), None)
                    current_detail = local_details_base.get(q, {})
                    candidate_detail = local_details_test.get(q, {})
                    # Give the query the larger impact observed under either
                    # option so a candidate cannot hide a newly-created regression.
                    score = max(
                        current_detail.get("impact", 0.0),
                        candidate_detail.get("impact", 0.0),
                    )
                    current_degradation = current_detail.get("degradation", 0.0)
                    candidate_degradation = candidate_detail.get("degradation", 0.0)
                    
                    if favored_val == current_val:
                        score_current_total += score
                        print(
                            f"          [Support Current] {q}: Score={score:.4f} "
                            f"(CurrentDeg={current_degradation*100:.2f}%, "
                            f"CandidateDeg={candidate_degradation*100:.2f}%)"
                        )
                    elif favored_val == candidate_val:
                        score_candidate_total += score
                        print(
                            f"          [Support Candidate] {q}: Score={score:.4f} "
                            f"(CurrentDeg={current_degradation*100:.2f}%, "
                            f"CandidateDeg={candidate_degradation*100:.2f}%)"
                        )

                print(f"       -> Total Score Comparison: Current[{score_current_total:.4f}] vs Candidate[{score_candidate_total:.4f}]")

                if score_candidate_total > score_current_total:
                    should_switch = True
                    print(f"       -> WINNER: Candidate ({candidate_val})")
                else:
                    should_switch = False
                    print(f"       -> WINNER: Current ({current_val})")

            if should_switch:
                print(f"     *** Update Parameter: {param} = {candidate_val}")
                C_base[param] = candidate_val
                P_resolved_locally.add(param)
            else:
                print(f"     *** Keep Parameter: {param} = {current_val}")
                P_resolved_locally.add(param)

        P_conflict_remaining -= P_resolved_locally
        
        # Recalibration
        if P_resolved_locally:
            print("\n[System] Local resolution complete, recalibrating L_base...")
            L_base, q_lats_base = self.evaluate_workload(C_base, verbose_print=True)

        regression_score_base, _ = self._calculate_workload_score(
            q_lats_base,
            q_lats_default,
            global_max_latency,
            global_max_freq,
            verbose_print=True,
            label="C_base before Global Greedy",
        )

        # --- Step 5: Global Greedy ---
        while P_conflict_remaining:
            print(f"\n--- Step 5: Global Greedy ({len(P_conflict_remaining)} remaining) ---")
            candidate_moves = []
            
            sorted_remaining = sorted(list(P_conflict_remaining), key=get_param_priority)
            
            for p in sorted_remaining:
                if p not in conflict_options: continue
                for v in conflict_options[p]:
                    if C_base.get(p) == v: continue
                    if (p, v) in self.timeout_blacklist: continue

                    print(f"  Evaluating: SET {p} = {v}")
                    C_test = C_base.copy()
                    C_test[p] = v
                    L_test, q_lats = self.evaluate_workload(C_test, verbose_print=True)
                    
                    if L_test == float('inf'):
                        self.timeout_blacklist.add((p, v))
                        continue
                    
                    self.log_data.append({
                        "Test_Type": "L_test (Greedy)", "Configuration": f"SET {p}={v}",
                        "Total_Latency_ms": L_test, **q_lats
                    })
                    
                    regression_score_test, _ = self._calculate_workload_score(
                        q_lats,
                        q_lats_default,
                        global_max_latency,
                        global_max_freq,
                        verbose_print=True,
                        label=f"Greedy Candidate {p}={v}",
                    )
                    self.log_data[-1]["Regression_Aware_Score"] = regression_score_test

                    raw_gain = L_base - L_test
                    weighted_gain = regression_score_base - regression_score_test
                    candidate_moves.append(
                        (p, v, weighted_gain, raw_gain, L_test, q_lats, regression_score_test)
                    )
                    print(
                        f"    Raw Gain: {raw_gain:.2f} ms | "
                        f"Regression-Aware Gain: {weighted_gain:.2f}"
                    )

            if not candidate_moves: break
            
            best_move = max(candidate_moves, key=lambda x: x[2])
            p_best, v_best, gain_best, raw_gain_best, L_best, q_lats_best, score_best = best_move
            
            if gain_best > 0:
                print(
                    f"*** Best move: {p_best} = {v_best} "
                    f"(Regression-Aware Gain: {gain_best:.2f}, Raw Gain: {raw_gain_best:.2f}ms)"
                )
                C_base[p_best] = v_best
                L_base = L_best
                q_lats_base = q_lats_best
                regression_score_base = score_best
            else:
                print(
                    f"*** Best move yields no positive regression-aware gain "
                    f"({gain_best:.2f}), keeping original value."
                )
            
            P_conflict_remaining.remove(p_best)

        print("\n--- Algorithm Complete (C_final) ---")
        return C_base

    def save_log_to_csv(self, filename: Path):
        if not self.log_data: return
        try:
            fieldnames = ["Test_Type", "Configuration", "Total_Latency_ms", "Regression_Aware_Score"]
            q_keys = [
                q for q in self.workload_q_map.keys()
                if any(q in row for row in self.log_data)
            ]
            fieldnames += q_keys
            
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                for row in self.log_data:
                    fmt_row = {k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in row.items()}
                    writer.writerow(fmt_row)
            print(f"Log saved: {filename}")
        except Exception as e:
            print(f"Error saving CSV: {e}")

# --- Main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dsn",
        default=os.getenv("PGRL_DSN", "application_name=PGRL-MERGE"),
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
    ap.add_argument("--report-dir", default="./training_log/experiment_reports", help="Directory containing the SQL reports to merge")
    ap.add_argument("--queries", default="Q1,Q2,Q3", help="Comma-separated list of queries to process")
    ap.add_argument(
        "--degradation-lambda",
        type=float,
        default=2.5,
        help="Regression-aware degradation multiplier lambda (default: 2.5)",
    )
    ap.add_argument(
        "--degradation-power",
        type=float,
        default=2.0,
        help="Regression-aware degradation exponent (default: 2.0, D squared)",
    )
    args = ap.parse_args()

    # Connection mode: remote SSH tunnel or local PostgreSQL in the same container.
    remote_mode = bool(args.ssh_host and args.ssh_host not in {"localhost", "127.0.0.1"})
    forwarder = None
    ssh_client = None
    merger = None

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
            print("[System] Running GCM in Local Mode (No SSH Tunnel).")
            forwarded_dsn = f"{args.dsn} host=127.0.0.1 port={args.remote_db_port} connect_timeout=10 sslmode=disable"
        
        # Prepare Workload
        target_qs = [q.strip() for q in args.queries.split(',') if q.strip()]
        workload_q_map = {q: SQL_MAP[q] for q in target_qs if q in SQL_MAP}
        all_tuned_params = list(PARAM_SPECS.keys())

        merger = ConfigMerger(
            dsn=forwarded_dsn,
            workload_q_map=workload_q_map,
            param_specs=PARAM_SPECS,
            all_tuned_params=all_tuned_params,
            ssh_client=ssh_client,
            ssh_password=args.ssh_password,
            report_dir=args.report_dir,
            remote_conf_path=args.remote_conf,
            degradation_lambda=args.degradation_lambda,
            degradation_power=args.degradation_power,
        )
        
        start_time = time.perf_counter()
        final_config = merger.run_merge_algorithm()
        duration = time.perf_counter() - start_time
        
        print(f"\nTotal algorithm duration: {duration/60:.1f} minutes")
        
        print("\n=========================================")
        print("Final converged parameters (C_final):")
        for k, v in final_config.items():
            print(f"  {k} = '{v}'")
        print("=========================================")
        
        # Final validation
        print("\n--- Validating C_final ---")
        for i in range(3):
            lat, q_lats = merger.evaluate_workload(final_config, verbose_print=True)
            print(f"Run {i+1}: {lat:.2f} ms")
            merger.log_data.append({"Test_Type": "Validation", "Configuration": f"Run {i+1}", "Total_Latency_ms": lat, **q_lats})

    except Exception as e:
        print(f"Execution error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if merger:
            print("\nRestoring system to original settings...")
            merger._update_remote_config({}) # Clear auto_tuning.conf
            merger._restart_remote_postgresql()
            
            # Prepare storage directory and timestamp
            log_dir = Path("./merge_test_log")
            log_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            #  Save C_final as an independent config file
            if 'final_config' in locals():
                final_conf_path = log_dir / f"C_final_{ts}.conf"
                try:
                    with open(final_conf_path, 'w', encoding='utf-8') as f:
                        f.write(f"# PGRL GCM Generated Configuration\n")
                        f.write(f"# Scoring Profile: {merger.scoring_profile}\n")
                        f.write(f"# Timestamp: {ts}\n")
                        f.write(f"# Algorithm Duration: {duration/60:.1f} minutes\n\n")
                        for k, v in final_config.items():
                            f.write(f"{k} = '{v}'\n")
                    print(f"Final configuration file (C_final) saved to: {final_conf_path}")
                except Exception as e:
                    print(f"Error saving C_final config file: {e}")

        if ssh_client:
            ssh_client.close()
        if forwarder and forwarder.is_active:
            forwarder.stop()

if __name__ == "__main__":
    main()
