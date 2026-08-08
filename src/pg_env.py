# pg_env_v16.py
# P1 and P2 Session/SET, P3 Restart hybrid environment
# Added Convergence Stopping Mechanism for P2/P3 phases
# Fixed Trust-Region Logic Bugs and Probe Handling of Boolean Parameters
# Added Support for Local Mode without SSH (e.g., Docker)
# Added catastrophic latency early stopping in callback
# v16: Modified latency measurement to use EXPLAIN ANALYZE with JSON output for more accurate timing and cost information, and to avoid issues with connection resets during long-running queries.

import gymnasium as gym
from gymnasium import spaces
import psycopg2
import time
import numpy as np
import paramiko
import subprocess
import os
import glob
import shlex
from typing import Sequence, Mapping
from stable_baselines3.common.callbacks import BaseCallback
from copy import deepcopy

from param_specs import PARAM_SPECS as DEFAULT_PARAM_SPECS

class PgConfEnv(gym.Env):

    metadata = {"render.modes": []}

    def __init__(self,
                dsn: str,
                # --- SSH ---
                ssh_client: paramiko.SSHClient = None,       
                ssh_password: str = None,                    
                remote_conf_path: str | None = None,
                
                # --- Env Config ---
                tuning_mode: str = "restart",  # "restart" or "session"
                tune_params=("work_mem",),
                workload=None,
                schedule: str = "single",
                episode_len: int = 32,
                param_specs: Mapping[str, dict] | None = None,
                
                # --- Control flags ---
                start_from_default: bool = True,
                baseline_first_step: bool = True,
                early_stop_factor: float | None = 5.0,
                min_timeout_ms: int = 3000,
                timeout_penalty: float = -100.0,
                
                # --- Trust-Region Control ---
                trim_mode: str = "auto",
                reset_trust_on_reset: bool = False,
                int_margin: int = 1,
                float_eps_rel: float = 0.25,
                float_eps_abs: float = 0.5,
                
                # --- Top-K ---
                topk_k: int = 5,
                
                # --- P1 converged params ---
                fixed_params: dict = None,
                initial_baseline_ms: float = None   # Phase 2 and Phase 3 can use this to set an initial latency baseline from Phase 1 results
                ):
        super().__init__()
        self.dsn = dsn
        self.tuning_mode = tuning_mode.lower()
        if self.tuning_mode not in ["restart", "session", "reload"]:
            raise ValueError("tuning_mode must be 'restart', 'session', or 'reload'")
        
        self.fixed_params = fixed_params if fixed_params else {}

        self.ssh_client = ssh_client
        self.ssh_password = ssh_password
        default_conf_path = os.path.join(os.getenv("PGDATA", "/var/lib/pgsql/data"), "auto_tuning.conf")
        self.remote_conf_path = remote_conf_path or os.getenv("REMOTE_CONF_PATH", default_conf_path)
        
        self.conn = None
        self._reconnect_db()

        self.start_from_default = bool(start_from_default)
        self.baseline_first_step = bool(baseline_first_step)
        self.latency_baseline_ms = float(initial_baseline_ms) if initial_baseline_ms is not None else None
        self.did_global_baseline = (initial_baseline_ms is not None)
        
        self.early_stop_factor = early_stop_factor
        self.min_timeout_ms = int(min_timeout_ms)
        self.timeout_penalty = float(timeout_penalty)
        self.trim_mode = trim_mode

        self.param_specs = dict(DEFAULT_PARAM_SPECS) if param_specs is None else dict(param_specs)

        self.tune_params = list(tune_params)
        for p in self.tune_params:
            assert p in self.param_specs, f"unknown param {p}"
        
        # P2 Session Parameters Dictionary
        self.current_session_params = {}

        # Trust-Region Control
        self.trust = {
            p: {"lo": float(self.param_specs[p]["min"]), "hi": float(self.param_specs[p]["max"])}
            for p in self.tune_params
        }
        self.trust_init = deepcopy(self.trust)
        self.reset_trust_on_reset = bool(reset_trust_on_reset)
        self.int_margin = int(int_margin)
        self.last_numeric_vals = {}
        self.last_trims = []
        self.float_eps_rel = float(float_eps_rel)
        self.float_eps_abs = float(float_eps_abs)
        
        # Best configuration b*
        self.best_latency_ms = None
        self.best_numeric = {}
        self.best_human = {}
        self.best_step = None
        self.best_query = None
        
        # Top-K & Probe
        self.topk_k = int(topk_k)
        self.last_topk = []
        self.probe_queue = []
        self.probe_override = None
        self.probe_current = None
        self.worst_numeric = None
        self.timeout_excluded = set()
        
        dim = len(self.tune_params)
        self.action_space = spaces.Box(-1.0, 1.0, (dim,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)

        self.workload_specs = workload or []
        if not self.workload_specs:
            raise ValueError("workload must be a non-empty list of QuerySpec objects")
        self.schedule = schedule
        self.episode_len = episode_len
        self._rr_idx = 0
        self._pick_active_query(first_time=True)

    # ---------- Connection & Config Helpers ----------
    def _local_pgdata(self) -> str:
        """Return the local PostgreSQL data directory used by Docker/local mode."""
        return os.getenv("PGDATA") or os.path.dirname(self.remote_conf_path)

    def _local_pg_ctl(self) -> str:
        """Resolve pg_ctl without hard-coding a PostgreSQL major version."""
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

    def _write_local_config(self, config_content: str):
        os.makedirs(os.path.dirname(self.remote_conf_path), exist_ok=True)
        with open(self.remote_conf_path, "w", encoding="utf-8") as f:
            f.write(config_content)
        subprocess.run(["chown", "postgres:postgres", self.remote_conf_path], check=True)
        subprocess.run(["chmod", "644", self.remote_conf_path], check=True)

    def _run_local_pg_ctl(self, action: str):
        pg_ctl = self._local_pg_ctl()
        pgdata = self._local_pgdata()
        wait_opt = " -w" if action in {"start", "stop", "restart"} else ""
        command = f"{shlex.quote(pg_ctl)} -D {shlex.quote(pgdata)}{wait_opt} {shlex.quote(action)}"
        subprocess.run(["su", "-", "postgres", "-c", command], check=True)

    def _update_remote_config_and_reload(self, params: dict[str, str]):
        """Apply config and reload PostgreSQL in either local or SSH mode."""
        combined_params = self.fixed_params.copy()
        combined_params.update(params)

        config_lines = [f"{k} = '{v}'" for k, v in combined_params.items()]
        config_content = "\n".join(config_lines)

        if self.ssh_client is None:
            try:
                print(f"[System] Applying config locally to {self.remote_conf_path}...")
                self._write_local_config(config_content)
                print("[System] Reloading Local PostgreSQL...")
                self._run_local_pg_ctl("reload")
                time.sleep(0.5)
                self._reconnect_db()
                return
            except Exception as e:
                print(f"[Error] Local Config Reload Failed: {e}")
                raise

        temp_path = "/tmp/auto_tuning.tmp"
        try:
            sftp = self.ssh_client.open_sftp()
            with sftp.file(temp_path, "w") as f:
                f.write(config_content)
            sftp.close()

            mv_cmd = (
                f"sudo -S mv -f {temp_path} {self.remote_conf_path} && "
                f"sudo -S chown postgres:postgres {self.remote_conf_path} && "
                f"sudo -S chmod 644 {self.remote_conf_path}"
            )
            stdin, stdout, stderr = self.ssh_client.exec_command(mv_cmd, get_pty=True)
            stdin.write((self.ssh_password or "") + "\n")
            stdin.flush()

            exit_code = stdout.channel.recv_exit_status()
            if exit_code != 0:
                out_msg = stdout.read().decode().strip()
                err_msg = stderr.read().decode().strip()
                raise RuntimeError(f"Failed to update config. Code: {exit_code}, STDOUT: {out_msg}, STDERR: {err_msg}")

            reload_cmd = "sudo -S systemctl reload postgresql"
            stdin, stdout, stderr = self.ssh_client.exec_command(reload_cmd, get_pty=True)
            stdin.write((self.ssh_password or "") + "\n")
            stdin.flush()

            exit_code = stdout.channel.recv_exit_status()
            if exit_code != 0:
                out_msg = stdout.read().decode().strip()
                err_msg = stderr.read().decode().strip()
                raise RuntimeError(f"PostgreSQL failed to reload: {out_msg} {err_msg}")

            time.sleep(0.5)
            self._reconnect_db()
        except Exception as e:
            print(f"SSH/Config Reload Error: {e}")
            raise

    def _reconnect_db(self):
        if self.conn:
            try:
                self.conn.close()
            except:
                pass
        
        max_retries = 10 if self.tuning_mode == "restart" else 3
        
        for attempt in range(max_retries):
            try:
                self.conn = psycopg2.connect(self.dsn)
                self.conn.autocommit = False 
                return
            except psycopg2.OperationalError:
                time.sleep(1.0)
        
        raise Exception(f"Failed to connect to DB mode={self.tuning_mode}")

    # Update config and restart logic for P3, with retry mechanism and better error handling
    def _update_remote_config_and_restart(self, params: dict[str, str], max_retries=3):
        full_config = self.fixed_params.copy()
        full_config.update(params)

        config_lines = [f"{k} = '{v}'" for k, v in full_config.items()]
        config_content = "\n".join(config_lines)
        
        if self.ssh_client is None:
            # === Local Mode (Docker) ===
            print(f"[System] Applying config locally to {self.remote_conf_path}...")
            try:
                self._write_local_config(config_content)
                print("[System] Restarting Local PostgreSQL...")
                self._run_local_pg_ctl("restart")
                print("[System] Local PostgreSQL restarted successfully.")
                time.sleep(2.0)
                self._reconnect_db()
                return
            except Exception as e:
                print(f"[Error] Local Restart Failed: {e}")
                raise
        
        else:
            # === SSH Mode (Original Logic) ===
            temp_path = "/tmp/auto_tuning.tmp"
            try:
                sftp = self.ssh_client.open_sftp()
                with sftp.file(temp_path, 'w') as f:
                    f.write(config_content)
                sftp.close()
                
                mv_cmd = f"sudo -S mv -f {temp_path} {self.remote_conf_path} && sudo -S chown postgres:postgres {self.remote_conf_path} && sudo -S chmod 644 {self.remote_conf_path}"
                stdin, stdout, stderr = self.ssh_client.exec_command(mv_cmd, get_pty=True)
                stdin.write(self.ssh_password + '\n')
                stdin.flush()
                
                exit_code = stdout.channel.recv_exit_status()
                
                if exit_code != 0:
                    out_msg = stdout.read().decode().strip()
                    err_msg = stderr.read().decode().strip()
                    
                    raise Exception(f"Failed to update config. Code: {exit_code}, STDOUT: '{out_msg}', STDERR: '{err_msg}'")

                restart_success = False
                last_error = ""
                for attempt in range(1, max_retries + 1):
                    restart_cmd = 'sudo -S systemctl restart postgresql'
                    stdin, stdout, stderr = self.ssh_client.exec_command(restart_cmd, get_pty=True)
                    stdin.write(self.ssh_password + '\n')
                    stdin.flush()
                    
                    if stdout.channel.recv_exit_status() == 0:
                        restart_success = True
                        break
                    else:
                        last_error = stderr.read().decode().strip()
                        print(f"[System] Restart failed ({attempt}): {last_error}")
                        if attempt < max_retries: time.sleep(3.0)

                if not restart_success:
                    raise RuntimeError(f"PostgreSQL failed to start: {last_error}")
                
                time.sleep(2.0) 
                self._reconnect_db()

            except Exception as e:
                print(f"SSH/Config Error: {e}")
                raise e

    def _apply_factory_defaults(self):
        if self.tuning_mode == "restart":
            self._update_remote_config_and_restart({})
        else:
            self.current_session_params = {}
            try:
                if self.conn and self.conn.closed == 0:
                    with self.conn.cursor() as cur:
                        cur.execute("RESET ALL;")
                    self.conn.commit()
            except Exception as e:
                print(f"[Warning] Failed to RESET ALL in session mode: {e}")
                self._reconnect_db()

    def _pick_active_query(self, first_time=False):
        if self.schedule == "single" and not first_time: return
        if self.schedule == "round_robin":
            self.active = self.workload_specs[self._rr_idx % len(self.workload_specs)]
            self._rr_idx += 1
        elif self.schedule == "random":
            import random; self.active = random.choice(self.workload_specs)
        else:
            self.active = self.workload_specs[0]
        self.sql, self.sql_params = self.active.sql, self.active.params

    # ---------- Helpers for TR logic ---------- 
    def _rank_all_by_deviation(self, cur_numeric: dict) -> list[dict]:
        if not self.best_numeric: return []
        items = []
        for p in self.tune_params:
            lo, hi = self.trust[p]["lo"], self.trust[p]["hi"]
            if hi - lo <= 0: continue
            if p not in cur_numeric or p not in self.best_numeric: continue
            width = max(hi - lo, 1e-9)
            cur_v, best_v = float(cur_numeric[p]), float(self.best_numeric[p])
            delta = cur_v - best_v
            is_bool = self._is_bool_param(p)
            items.append({
                "param": p, "is_bool": is_bool, "delta": delta,
                "norm": delta / width, "abs_norm": abs(delta) / width,
                "cur": cur_v, "best": best_v, "lo": lo, "hi": hi,
            })
        items.sort(key=lambda d: d["abs_norm"], reverse=True)
        items.sort(key=lambda d: (not d["is_bool"]))
        return items

    def _fix_trust_bounds(self, p: str):
        spec = self.param_specs[p]
        lo = max(float(spec["min"]), min(self.trust[p]["lo"], float(spec["max"])))
        hi = max(float(spec["min"]), min(self.trust[p]["hi"], float(spec["max"])))
        if lo > hi: m = 0.5 * (lo + hi); lo = hi = m
        self.trust[p]["lo"], self.trust[p]["hi"] = lo, hi    
    
    def _fmt_range(self, p: str) -> str:
        lo, hi = self.trust[p]["lo"], self.trust[p]["hi"]
        sample = self.param_specs[p]["cast"]((lo + hi) / 2.0)
        if isinstance(sample, (int, str)): return f"{int(lo)}..{int(hi)}"
        return f"{lo:.3g}..{hi:.3g}"
    
    def _is_bool_param(self, p: str) -> bool:
        spec = self.param_specs[p]
        if spec.get("is_bool", False):
            return True
        # Backward-compatible detection for older spec files.
        if spec.get("min") == 0 and spec.get("max") == 1 and callable(spec.get("fmt")):
            try:
                return {str(spec["fmt"](0)).lower(), str(spec["fmt"](1)).lower()} == {"off", "on"}
            except Exception:
                pass
        return False

    def _setting_to_mb(self, setting: str, unit: str | None) -> float:
        value = float(setting)
        if value == -1:
            return -1.0
        if not unit:
            return value
        if unit.endswith("kB"):
            prefix = unit[:-2]
            multiplier = int(prefix) if prefix.isdigit() else 1
            return value * multiplier / 1024.0
        if unit == "MB":
            return value
        if unit == "GB":
            return value * 1024.0
        return value

    def _to_numeric_from_pg(self, p: str, setting: str, unit: str | None) -> float:
        """Convert pg_settings output into the same numeric domain used by PPO/TRIM.

        For level-based parameters the RL domain is the level index, not the
        physical PostgreSQL value. Keeping this representation consistent is
        essential for b*, timeout ranking, probes, and trust-region shrinking.
        """
        spec = self.param_specs[p]
        levels = spec.get("levels")

        if levels:
            try:
                if float(setting) == -1 and -1 in levels:
                    actual_value = -1.0
                elif spec.get("levels_unit") == "MB":
                    actual_value = self._setting_to_mb(setting, unit)
                else:
                    actual_value = float(setting)

                nearest_index = min(
                    range(len(levels)),
                    key=lambda i: abs(float(levels[i]) - actual_value),
                )
                return float(nearest_index)
            except Exception:
                return float(spec.get("min", 0))

        if unit is None or unit == "":
            s = setting.strip().lower()
            if s in ("on", "off"):
                return 1.0 if s == "on" else 0.0
            try:
                return float(setting)
            except Exception:
                return 0.0

        if str(setting).strip() == "-1":
            return -1.0
        if unit.endswith("kB"):
            try:
                return self._setting_to_mb(setting, unit)
            except Exception:
                return 0.0
        try:
            return float(setting)
        except Exception:
            return 0.0

    def _record_trim(self, p: str, old_lo: float, old_hi: float, new_lo: float, new_hi: float, v: float, mode: str, eps: float):
        if (old_lo, old_hi) != (new_lo, new_hi):
            self.last_trims.append(
                f"{p}:{mode} @{v:.3g} (−{eps:.3g}) {int(old_lo) if float(old_lo).is_integer() else old_lo:.3g}.."
                f"{int(old_hi) if float(old_hi).is_integer() else old_hi:.3g} → "
                f"{int(new_lo) if float(new_lo).is_integer() else new_lo:.3g}.."
                f"{int(new_hi) if float(new_hi).is_integer() else new_hi:.3g}"
            )

    def _fmt_num_for_log(self, p: str, x: float) -> str:
        spec = self.param_specs[p]
        if spec.get("levels"):
            try:
                val = spec["cast"](x)
                formatter = spec.get("fmt", "{val}")
                return str(formatter(val) if callable(formatter) else formatter.format(val=val))
            except Exception:
                pass
        lo, hi = self.trust[p]["lo"], self.trust[p]["hi"]
        sample = spec["cast"]((lo + hi) / 2.0)
        if isinstance(sample, (int, str)):
            try:
                return str(int(round(x)))
            except Exception:
                return f"{x:.3g}"
        return f"{x:.3g}"

    def _schedule_probes_from_topk(self, topk: list[dict], bad_numeric: dict | None = None):
        self.probe_queue = []
        if not topk or not self.best_numeric: return
        for it in topk:
            p = it["param"]
            lo, hi = self.trust[p]["lo"], self.trust[p]["hi"]
            target = float(self.best_numeric.get(p, it["cur"]))
            probe = min(max(target, lo), hi)
            self.probe_queue.append({
                "param": p, "probe_numeric": probe, "target_best": target, "clamped": (probe != target),
                "bad_numeric": None if bad_numeric is None else bad_numeric.get(p, None), "from_timeout": True,
            })

    def _shrink_trust_region_on_probe_success(self, pr: dict):
        p = pr["param"]
        good = float(pr["probe_numeric"])
        bad  = pr.get("bad_numeric", None)
        lo, hi = self.trust[p]["lo"], self.trust[p]["hi"]
        old_lo, old_hi = lo, hi
        
        spec = self.param_specs[p]
        sample = spec["cast"]((lo + hi) / 2.0)
        
        is_boolean_like = self._is_bool_param(p)

        if is_boolean_like:
            self.trust[p]["lo"] = self.trust[p]["hi"] = good
            self._fix_trust_bounds(p)
            self._record_trim(p, old_lo, old_hi, self.trust[p]["lo"], self.trust[p]["hi"], good, "probe→pin", 0.0)
            return
        
        if bad is None:
            shrink_right = (abs(hi - good) >= abs(good - lo))
            eps_base = max(self.float_eps_abs, self.float_eps_rel * float(hi - lo))
        else:
            shrink_right = (bad > good)
            eps_base = max(self.float_eps_abs, self.float_eps_rel * abs(float(bad - good)))
        cast = self.param_specs[p]["cast"]
        if shrink_right:
            used_eps = min(eps_base, max(0.0, float(hi - good)))
            new_lo, new_hi = lo, cast(float(hi) - used_eps)
            if new_hi < good: new_hi = cast(good)
        else:
            used_eps = min(eps_base, max(0.0, float(good - lo)))
            new_lo, new_hi = cast(float(lo) + used_eps), hi
            if new_lo > good: new_lo = cast(good)
        self.trust[p]["lo"], self.trust[p]["hi"] = new_lo, new_hi
        self._fix_trust_bounds(p)
        self._record_trim(p, old_lo, old_hi, self.trust[p]["lo"], self.trust[p]["hi"], good, 
                        "probe→proportional-one-sided" + (">" if shrink_right else "<"), float(used_eps))
    
    def _maybe_update_bstar(self, *, timed_out: bool, lat_ms: float,
                            cur_human: dict, cur_numeric: dict, step_idx: int, query_name: str):
        if timed_out: return False
        if (self.best_latency_ms is None) or (lat_ms < self.best_latency_ms):
            self.best_latency_ms = float(lat_ms)
            self.best_numeric = dict(cur_numeric)
            self.best_human = dict(cur_human)
            self.best_step = int(step_idx)
            self.best_query = query_name
            return True
        return False
    
    def _execute_sql(self, timeout_ms: int | None = None):
        sql, params = self.sql, self.sql_params
        
        for attempt in (1, 2):
            try:
                with self.conn.cursor() as cur:
                    if self.tuning_mode == "session" and self.current_session_params:
                        for k, v in self.current_session_params.items():
                            try:
                                cur.execute(f"SET {k} = '{v}'")
                            except Exception as set_e:
                                print(f"[Warning] Failed to SET {k}={v}: {set_e}")

                    if timeout_ms is not None:
                        cur.execute("SET statement_timeout = %s", (int(timeout_ms),))
                    
                    try:
                        # Use EXPLAIN ANALYZE with JSON format to get detailed timing information
                        explain_sql = "EXPLAIN (ANALYZE, FORMAT JSON) " + sql
                        cur.execute(explain_sql, params)
                        explain_plan = cur.fetchone()[0][0]
                        
                        # Extract Planning Time and Execution Time from the JSON output
                        planning_time = explain_plan.get("Planning Time", 0.0)
                        execution_time = explain_plan.get("Execution Time", 0.0)
                        total_cost = explain_plan["Plan"].get("Total Cost", 0.0)
                        
                        # Total latency is the sum of both times
                        lat = planning_time + execution_time
                        
                        if timeout_ms is not None: cur.execute("SET statement_timeout = 0")
                        return lat, False, total_cost
                    except Exception as e:
                        msg = str(e)
                        if "statement timeout" in msg or "canceling statement" in msg or getattr(e, "pgcode", None) == "57014":
                            try: self.conn.rollback()
                            except: pass
                            try:
                                with self.conn.cursor() as c2: c2.execute("SET statement_timeout = 0")
                            except: pass
                            
                            lat = float(timeout_ms if timeout_ms is not None else self.min_timeout_ms)
                            if lat <= 0: lat = self.min_timeout_ms
                            
                            # Get the cost from EXPLAIN when timeout occurs, since we won't have actual execution time. 
                            # This can help the agent learn even from timeouts.
                            real_timeout_cost = 1e20
                            try:
                                with self.conn.cursor() as c3:
                                    c3.execute("EXPLAIN (FORMAT JSON) " + sql, params)
                                    real_timeout_cost = c3.fetchone()[0][0]["Plan"]["Total Cost"]
                            except Exception as explain_e:
                                print(f"[Warning] Failed to get real cost during timeout: {explain_e}")
                                
                            return lat, True, real_timeout_cost
                        raise
                    
            except psycopg2.OperationalError as e:
                msg = str(e)
                if attempt == 1 and ("SSL SYSCALL" in msg or "reset by peer" in msg or "connection not open" in msg or "closed" in msg):
                    print(f"[Warning] _execute_sql found closed connection (mode={self.tuning_mode}), reconnecting...")
                    self._reconnect_db()
                    continue
                raise

    def _total_cost(self):
        try:
            with self.conn.cursor() as cur: cur.execute("ROLLBACK")
        except:
            try: self.conn.rollback()
            except: pass
            
        with self.conn.cursor() as cur:
            if self.tuning_mode == "session" and self.current_session_params:
                for k, v in self.current_session_params.items():
                    cur.execute(f"SET {k} = '{v}'")
                    
            cur.execute("EXPLAIN (FORMAT JSON) " + self.sql, self.sql_params)
            return cur.fetchone()[0][0]["Plan"]["Total Cost"]

    def _obs(self):
        cost = self._total_cost()
        return np.array([np.log1p(cost)], dtype=np.float32), cost

    # ---------- gym API ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.reset_trust_on_reset:
            self.trust = deepcopy(self.trust_init)
        self.last_trims = []
        self.step_cnt = 0
        self._pick_active_query()
        
        # Decide whether to reset to factory defaults at the start of each episode
        if self.start_from_default:
            self._apply_factory_defaults()
        
        if self.tuning_mode == "session":
            self.current_session_params = {}

        obs, _ = self._obs()
        return obs, {}

    def step(self, action):
        self.last_trims = []
        restart_failed = False
        params_to_apply = {} 

        # 1. Baseline logic
        if self.baseline_first_step and self.step_cnt == 0 and (not self.did_global_baseline):
            try:
                self._apply_factory_defaults()
            except Exception as e:
                restart_failed = True
                print(f"[Env] Crash detected! {e}")
            timeout_ms = None
        else:
            self.probe_override = None
            if self.probe_current is None and self.probe_queue:
                self.probe_current = self.probe_queue.pop(0)
            
            # Calculate parameters
            self.last_numeric_vals = {} 
            for idx, p in enumerate(self.tune_params):
                spec = self.param_specs[p]
                reg  = self.trust[p]
                raw  = float(np.clip(action[idx], -1.0, 1.0))

                # 1. Check for Probe Override
                override_val = None
                if self.probe_override and p in self.probe_override:
                    override_val = float(self.probe_override[p])
                elif self.probe_current is not None and self.probe_current["param"] == p:
                    override_val = float(self.probe_current["probe_numeric"])
                
                # 2. Calculate target value (val)
                # If there is a Probe request, use the Probe value 
                # Otherwise, use the Agent's Action to calculate
                if override_val is not None:
                    val = spec["cast"](override_val)
                else:
                    cont = reg["lo"] + (raw + 1.0) * 0.5 * (reg["hi"] - reg["lo"])
                    val  = spec["cast"](cont)
                
                # 3. Handle virtual parameters vs regular parameters
                if spec.get("is_virtual", False):
                    # If it is a virtual parameter, expand this single value (val) to all target parameters
                    target_params = spec.get("map_to", [])
                    
                    # Use the virtual parameter's formatter to format the value
                    formatter = spec["fmt"]
                    if callable(formatter):
                        literal = formatter(val)
                    else:
                        literal = formatter.format(val=val)
                    
                    # Assign to all corresponding real parameters
                    for target_p in target_params:
                        params_to_apply[target_p] = str(literal)
                        self.last_numeric_vals[target_p] = float(val)
                    
                    # Also record the virtual parameter itself for debugging
                    self.last_numeric_vals[p] = float(val)

                else:
                    # Regular parameter handling logic
                    formatter = spec["fmt"]
                    if callable(formatter):
                        literal = formatter(val)
                    else:
                        literal = formatter.format(val=val)
                    
                    params_to_apply[p] = str(literal)

                    if isinstance(val, str):
                        self.last_numeric_vals[p] = 1 if val == 'on' else 0
                    else:
                        self.last_numeric_vals[p] = float(val)
            
            # Apply Parameters based on Mode
            if self.tuning_mode == "restart":
                # Case 1: Write Config + Restart (e.g., shared_buffers and max_worker_processes)
                try:
                    self._update_remote_config_and_restart(params_to_apply)
                except Exception as e:
                    restart_failed = True
                    print(f"[Env] Crash detected (Restart)! {params_to_apply}")
            
            elif self.tuning_mode == "reload":
                # Case 2: Write Config + Reload (e.g., autovacuum_work_mem)
                try:
                    self._update_remote_config_and_reload(params_to_apply)

                    self.current_session_params = {} 
                except Exception as e:
                    restart_failed = True
                    print(f"[Env] Crash detected (Reload)! {params_to_apply}")        
            
            else:
                # Case 3: Session Mode - SET parameters for current session
                # No restart needed
                self.current_session_params = params_to_apply

            timeout_ms = None
            if (self.latency_baseline_ms is not None) and (self.early_stop_factor is not None):
                timeout_ms = max(self.min_timeout_ms, int(self.latency_baseline_ms * float(self.early_stop_factor)))

        # 2. Crash Handling
        if restart_failed:
            # (Crash logic remains same ...)
            obs = np.array([20.0], dtype=np.float32)
            return obs, self.timeout_penalty * 10, True, False, {
                "latency_ms": timeout_ms * 2 if timeout_ms else 10000, 
                "total_cost": 1e9, "error": "db_crash"
            }

        # 3. Execute SQL
        lat_ms, timed_out, cost = self._execute_sql(timeout_ms=timeout_ms)
        
        # 4. Reward & Post-processing
        if timed_out:
            if self.probe_current is None:
                self.worst_numeric = dict(self.last_numeric_vals)
                ranked = self._rank_all_by_deviation(self.worst_numeric)

                if ranked and ranked[0]["is_bool"] and abs(ranked[0]["delta"]) > 0.5:
                    topk = [it for it in ranked if it["is_bool"]]
                    print(f"[TRIM Protection] Timeout caused by Flags. Scheduling only Boolean probes.")
                else:
                    topk = ranked[: self.topk_k]
                
                self.last_topk = topk
                if topk:
                    self._schedule_probes_from_topk(topk, bad_numeric=self.worst_numeric)
            else:
                self.timeout_excluded.add(self.probe_current["param"])
        else:
            if self.probe_current is not None:
                self._shrink_trust_region_on_probe_success(self.probe_current)
                self.probe_queue = []
                self.worst_numeric = None
                self.timeout_excluded = set()
            
            if self.step_cnt == 0 and self.latency_baseline_ms is None:
                self.latency_baseline_ms = float(lat_ms)
                self.did_global_baseline = True

        obs = np.array([np.log1p(cost)], dtype=np.float32)

        if self.latency_baseline_ms is None:
            # A baseline execution can fail before a usable latency is established.
            # Return a penalty instead of attempting arithmetic with None.
            reward = self.timeout_penalty if timed_out else -np.log(lat_ms + 1e-6)
        elif lat_ms / (self.latency_baseline_ms + 1e-6) > 1:
            reward = (lat_ms / (self.latency_baseline_ms + 1e-6)) * self.timeout_penalty
        else:
            reward = -np.log(lat_ms + 1e-6)

        self.step_cnt += 1
        done  = self.step_cnt >= self.episode_len
        trunc = False

        info = {"latency_ms": lat_ms, "total_cost": cost}
        if self.latency_baseline_ms is not None:
            info["latency_baseline_ms"] = self.latency_baseline_ms
        if timeout_ms is not None:
            info["timeout_ms"] = timeout_ms
            info["early_stop"] = bool(timed_out)

        cur_human, cur_numeric = {}, {}
        with self.conn.cursor() as cur:
            for p in self.tune_params:
                spec = self.param_specs[p]
                
                # Handle virtual parameters
                if spec.get("is_virtual", False):
                    # The value of a virtual parameter is determined by the last action and stored in self.last_numeric_vals when we applied the parameters.
                    # We need to retrieve it from there instead of querying pg_settings.
                    raw_val = self.last_numeric_vals.get(p, 0)
                    
                    # Cast it to the appropriate type using the spec's cast function (defaulting to float if not specified).
                    cast_func = spec.get("cast", float)
                    val = cast_func(raw_val)
                    
                    # Format the value
                    formatter = spec.get("fmt", "{val}")
                    if callable(formatter):
                        human_val = str(formatter(val))
                    else:
                        human_val = str(formatter.format(val=val))
                    
                    # Record the virtual parameter itself in the info dictionary
                    info[p] = human_val
                    cur_human[p] = human_val
                    cur_numeric[p] = float(val)
                    
                    target_params = spec.get("map_to", [])
                    for tp in target_params:
                        cur.execute("SELECT setting, unit FROM pg_settings WHERE name = %s", (tp,))
                        res = cur.fetchone()
                        if res:
                            setting, unit = res
                            human = self._humanize_setting(setting, unit)
                            info[tp] = human
                            
                # Regular parameters are retrieved directly from pg_settings
                else:
                    cur.execute("SELECT setting, unit FROM pg_settings WHERE name = %s", (p,))
                    res = cur.fetchone()
                    if res:
                        setting, unit = res
                        human = self._humanize_setting(setting, unit)
                        info[p] = human
                        cur_human[p] = human
                        cur_numeric[p] = self._to_numeric_from_pg(p, setting, unit)

        for p in self.tune_params:
            info[f"TR_{p}"] = self._fmt_range(p)
        if self.last_trims:
            info["TRIM"] = "; ".join(self.last_trims)
        
        if self.probe_current is not None:
            p = self.probe_current["param"]
            info["PROBE_policy"] = "one-at-a-time toward b*"
            info["PROBE_param"]  = p
            info["PROBE_value"]  = self._fmt_num_for_log(p, self.probe_current["probe_numeric"])
        
        updated_bstar = self._maybe_update_bstar(
            timed_out=timed_out, lat_ms=lat_ms, cur_human=cur_human, cur_numeric=cur_numeric,
            step_idx=self.step_cnt, query_name=getattr(getattr(self, "active", None), "name", None),
        )
        if updated_bstar:
            info["B*_updated"] = True
        
        self.probe_current = None
        return obs, reward, done, trunc, info

    def _humanize_setting(self, setting: str, unit: str | None) -> str:
        setting = str(setting).strip()
        if setting == "-1": return "-1"
        if not unit: return setting
        try:
            if unit.endswith("kB"):
                mul = 1
                if unit != "kB":
                    prefix = unit[:-2]
                    if prefix.isdigit(): mul = int(prefix)
                kb = int(float(setting)) * mul
                mb = kb / 1024.0
                return f"{int(mb)}MB" if mb.is_integer() else f"{mb:.2f}MB"
            return f"{setting}{unit}"
        except: return setting

class PPOLogger(BaseCallback):
    def __init__(self, print_header: bool = True):
        super().__init__()
        self.print_header = print_header
        self._ep_count = 0
    def _env0(self):
        try:
            venv = self.training_env
            env = venv.envs[0]
            while hasattr(env, "env"): env = env.env
            return getattr(env, "unwrapped", env)
        except: return None
    def _on_training_start(self) -> None:
        if not self.print_header: return
        e = self._env0()
        if e is None: return
        # [NEW] Log the mode
        print(f"[RUNCFG] mode={getattr(e, 'tuning_mode', 'unknown')}, dsn={getattr(e, 'dsn', None)}", flush=True)
    def _on_rollout_start(self) -> None:
        self._ep_count += 1
        print(f"[EP {self._ep_count}] Start...", flush=True)
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if not infos: return True
        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        info.pop("TimeLimit.truncated", None)
        step  = self.num_timesteps
        print(f"[{step:>6}] " + ", ".join(f"{k}={v}" for k, v in info.items()), flush=True)
        return True

# When the environment's best latency does not improve by more than 'min_delta' for 'patience' steps, stop training.
# Includes a 'warmup_steps' period to ignore initial "fresh start" outliers.
class ConvergenceStoppingCallback(BaseCallback):
    def __init__(self, patience: int = 200, min_delta_ratio: float = 0.01, check_freq: int = 1, warmup_steps: int = 20, verbose: int = 1, catastrophic_patience: int = 20):
        super().__init__(verbose)
        self.patience = patience
        self.min_delta_ratio = min_delta_ratio
        self.check_freq = check_freq
        self.warmup_steps = warmup_steps
        
        self.catastrophic_patience = catastrophic_patience
        self.consecutive_timeouts = 0
        
        self.best_latency = np.inf
        self.wait_count = 0
        self.last_checked_step = 0

    def _env_unwrapped(self):
        env = self.training_env.envs[0]
        while hasattr(env, "env"):
            env = env.env
        return getattr(env, "unwrapped", env)

    def _on_step(self) -> bool:
        # Skip checks during warmup period
        if self.num_timesteps < self.warmup_steps:
            return True

        if self.n_calls % self.check_freq != 0:
            return True

        env = self._env_unwrapped()
        
        # Retrieve current step latency from infos
        infos = self.locals.get("infos", [{}])[0]
        current_step_latency = infos.get("latency_ms", getattr(env, "best_latency_ms", None))
        
        is_timeout = infos.get("early_stop", False)

        if is_timeout:
            self.consecutive_timeouts += self.check_freq
            if self.consecutive_timeouts >= self.catastrophic_patience:
                if self.verbose > 0:
                    print(f"\n[EarlyStop] Catastrophic! Hit {self.catastrophic_patience} consecutive timeouts.")
                    print("[EarlyStop] The agent is trapped in a terrible parameter space. Force stopping phase.")
                return False
        else:
            self.consecutive_timeouts = 0
            
        if current_step_latency is None:
            return True
        
        if self.best_latency == np.inf:
            self.best_latency = current_step_latency
            if self.verbose > 0:
                print(f"[EarlyStop] Warmup done. Benchmark initialized at {self.best_latency:.2f} ms")
            return True

        # Calculate improvement based on CURRENT step vs Historical Best
        improvement = self.best_latency - current_step_latency
        threshold = self.best_latency * self.min_delta_ratio

        if improvement > threshold:
            if self.verbose > 0:
                print(f"[EarlyStop] Improved! {self.best_latency:.2f} -> {current_step_latency:.2f} (Delta: {improvement:.2f} > {threshold:.2f})")
            self.best_latency = current_step_latency
            self.wait_count = 0 
        else:
            self.wait_count += self.check_freq
            
        if self.wait_count >= self.patience:
            if self.verbose > 0:
                print(f"\n[EarlyStop] Stopping training! No improvement for {self.wait_count} steps.")
                print(f"[EarlyStop] Best Latency stalled at {self.best_latency:.2f} ms")
            return False
            
        return True

__all__ = ["PgConfEnv", "PPOLogger", "ConvergenceStoppingCallback"]
