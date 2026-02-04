# pg_env_v14_hierarchical.py
# P1 and P2 Session/SET, P3 Restart hybrid environment
# Added Convergence Stopping Mechanism for P2/P3 phases
# Fixed Trust-Region Logic Bugs and Probe Handling of Boolean Parameters
# Added Support for Local Mode without SSH (e.g., Docker)

import gymnasium as gym
from gymnasium import spaces
import psycopg2
import time
import numpy as np
import paramiko
import subprocess
from typing import Sequence, Mapping
from stable_baselines3.common.callbacks import BaseCallback
from copy import deepcopy

# 預設載入全部，但建議由外部傳入 P1_PARAM_SPECS 或 P2_PARAM_SPECS
from param_specs import PARAM_SPECS as DEFAULT_PARAM_SPECS

class PgConfEnv(gym.Env):

    metadata = {"render.modes": []}

    def __init__(self,
                dsn: str,
                # --- SSH 相關 (P1 必須，P2 可選但建議保留以防萬一要 reset config) ---
                ssh_client: paramiko.SSHClient = None,       
                ssh_password: str = None,                    
                remote_conf_path: str = "/var/lib/pgsql/data/auto_tuning.conf",
                
                # --- 環境設定 ---
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
                fixed_params: dict = None
                ):
        super().__init__()
        self.dsn = dsn
        self.tuning_mode = tuning_mode.lower()
        if self.tuning_mode not in ["restart", "session", "reload"]:
            raise ValueError("tuning_mode must be 'restart', 'session', or 'reload'")
        
        self.fixed_params = fixed_params if fixed_params else {}

        # SSH 檢查: Restart 模式必須有 SSH
        # if self.tuning_mode == "restart":
        #     if not ssh_client or not ssh_password:
        #         raise ValueError("Restart mode requires ssh_client and ssh_password")

        self.ssh_client = ssh_client
        self.ssh_password = ssh_password
        self.remote_conf_path = remote_conf_path
        
        # 連線物件
        self.conn = None
        self._reconnect_db()

        self.start_from_default = bool(start_from_default)
        self.baseline_first_step = bool(baseline_first_step)
        self.did_global_baseline = False
        
        self.early_stop_factor = early_stop_factor
        self.min_timeout_ms = int(min_timeout_ms)
        self.timeout_penalty = float(timeout_penalty)
        self.latency_baseline_ms = None
        self.trim_mode = trim_mode

        self.param_specs = dict(DEFAULT_PARAM_SPECS) if param_specs is None else dict(param_specs)

        self.tune_params = list(tune_params)
        for p in self.tune_params:
            assert p in self.param_specs, f"unknown param {p}"
        
        # P2 模式專用：儲存當前的 session 參數 (key: value_str)
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
      
    def _update_remote_config_and_reload(self, params: dict[str, str]):
        """Combine P1 fixed (converged) params with current params, write config file, and reload (P2 Only)"""
        
        # 1. Combine params
        combined_params = self.fixed_params.copy()
        combined_params.update(params)

        config_lines = [f"{k} = '{v}'" for k, v in combined_params.items()]
        config_content = "\n".join(config_lines)
        temp_path = "/tmp/auto_tuning.tmp"
        
        try:
            # 2. SFTP upload to temp file
            sftp = self.ssh_client.open_sftp()
            with sftp.file(temp_path, 'w') as f:
                f.write(config_content)
            sftp.close()
            
            # 3. Move file (overwrite auto_tuning.conf)
            mv_cmd = f"sudo -S mv -f {temp_path} {self.remote_conf_path} && sudo -S chown postgres:postgres {self.remote_conf_path} && sudo -S chmod 644 {self.remote_conf_path}"
            stdin, stdout, stderr = self.ssh_client.exec_command(mv_cmd, get_pty=True)
            stdin.write(self.ssh_password + '\n')
            stdin.flush()
            
            exit_code = stdout.channel.recv_exit_status()
            if exit_code != 0:
                out_msg = stdout.read().decode().strip()
                err_msg = stderr.read().decode().strip()
                raise Exception(f"Failed to update config. Code: {exit_code}, STDOUT: {out_msg}, STDERR: {err_msg}")

            # 4. Execute Reload command (faster than Restart)
            reload_cmd = 'sudo -S systemctl reload postgresql'
            stdin, stdout, stderr = self.ssh_client.exec_command(reload_cmd, get_pty=True)
            stdin.write(self.ssh_password + '\n')
            stdin.flush()
            
            exit_code = stdout.channel.recv_exit_status()
            if exit_code != 0:
                out_msg = stdout.read().decode().strip()
                err_msg = stderr.read().decode().strip()
                raise RuntimeError(f"PostgreSQL failed to reload: {out_msg} {err_msg}")
            
            # 5. Wait a moment to ensure settings take effect (SIGHUP is asynchronous)
            time.sleep(0.5) 
            
            # Optional: Reload usually doesn't require reconnect, but to be safe and ensure the session picks up new settings, reconnecting is more reliable
            self._reconnect_db()

        except Exception as e:
            print(f"SSH/Config Reload Error: {e}")
            raise e
    
    def _reconnect_db(self):
        """建立或重新建立 DB 連線"""
        if self.conn:
            try:
                self.conn.close()
            except:
                pass
        
        # Session 模式通常不需要像 Restart 模式那樣 retry 這麼多次，除非網路不穩
        max_retries = 10 if self.tuning_mode == "restart" else 3
        
        for attempt in range(max_retries):
            try:
                self.conn = psycopg2.connect(self.dsn)
                self.conn.autocommit = False 
                return
            except psycopg2.OperationalError:
                time.sleep(1.0)
        
        raise Exception(f"Failed to connect to DB mode={self.tuning_mode}")

    def _update_remote_config_and_restart(self, params: dict[str, str], max_retries=3):
        """(P3 Only) 寫入設定檔並重啟"""
        
        # [修正] 合併固定參數 (P1/P2 Context) 與 當前嘗試參數 (P3 Action)
        # 這樣寫入設定檔時，才會包含所有階段的設定
        full_config = self.fixed_params.copy()
        full_config.update(params)

        # 使用 full_config 來產生設定檔內容
        config_lines = [f"{k} = '{v}'" for k, v in full_config.items()]
        config_content = "\n".join(config_lines)
        
        if self.ssh_client is None:
            # === Local Mode (Docker) ===
            print(f"[System] Applying config locally to {self.remote_conf_path}...")
            try:
                # 1. 寫入設定檔
                # 注意: Docker 內通常是以 root 或 postgres 執行，直接寫入即可
                # 如果權限不足，可能需要切換使用者，但在 entrypoint 我們已經給了權限
                with open(self.remote_conf_path, 'w') as f:
                    f.write(config_content)
                
                # 確保權限正確
                subprocess.run(f"chown postgres:postgres {self.remote_conf_path}", shell=True, check=True)
                
                # 2. 重啟 DB
                print("[System] Restarting Local PostgreSQL...")
                # 切換成 postgres 使用者執行 pg_ctl restart
                cmd = "su - postgres -c '/usr/lib/postgresql/15/bin/pg_ctl -D /var/lib/postgresql/data -w restart'"
                subprocess.run(cmd, shell=True, check=True)
                
                print("[System] Local PostgreSQL restarted successfully.")
                time.sleep(2.0)
                self._reconnect_db()
                return # 成功就返回

            except Exception as e:
                print(f"[Error] Local Restart Failed: {e}")
                raise e
        
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
                    # [修正] 當 get_pty=True 時，錯誤訊息通常在 stdout 而不是 stderr
                    # 我們把兩邊都讀出來，確保不會漏掉訊息
                    out_msg = stdout.read().decode().strip()
                    err_msg = stderr.read().decode().strip()
                    
                    # 組合完整的錯誤訊息
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
        """Reset Logic"""
        if self.tuning_mode == "restart":
            # P1: 清空設定檔並重啟
            self._update_remote_config_and_restart({})
        else:
            # P2: 不動設定檔，只清空當前的 session 參數記憶，並在 DB 執行 RESET ALL
            self.current_session_params = {}
            try:
                if self.conn and self.conn.closed == 0:
                    with self.conn.cursor() as cur:
                        cur.execute("RESET ALL;") # 回到 Config File 的設定 (即 P1 的結果)
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

    # ---------- Helpers for TR logic (Unchanged) ---------- 
    def _rank_all_by_deviation(self, cur_numeric: dict) -> list[dict]:
        # (保持原樣 ...)
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
        # (保持原樣 ...)
        spec = self.param_specs[p]
        lo = max(float(spec["min"]), min(self.trust[p]["lo"], float(spec["max"])))
        hi = max(float(spec["min"]), min(self.trust[p]["hi"], float(spec["max"])))
        if lo > hi: m = 0.5 * (lo + hi); lo = hi = m
        self.trust[p]["lo"], self.trust[p]["hi"] = lo, hi    
    
    def _fmt_range(self, p: str) -> str:
        # (保持原樣 ...)
        lo, hi = self.trust[p]["lo"], self.trust[p]["hi"]
        sample = self.param_specs[p]["cast"]((lo + hi) / 2.0)
        if isinstance(sample, (int, str)): return f"{int(lo)}..{int(hi)}"
        return f"{lo:.3g}..{hi:.3g}"
    
    def _is_bool_param(self, p: str) -> bool:
        lo, hi = self.trust[p]["lo"], self.trust[p]["hi"]
        sample = self.param_specs[p]["cast"]((lo + hi) / 2.0)
        return isinstance(sample, str)
    
    def _to_numeric_from_pg(self, p: str, setting: str, unit: str | None) -> float:
        # (保持原樣 ...)
        if unit is None or unit == "":
            s = setting.strip().lower()
            if s in ("on", "off"): return 1.0 if s == "on" else 0.0
            try: return float(setting)
            except: return 0.0
        if unit.endswith("kB"):
            mul = 1
            prefix = unit[:-2]
            if prefix.isdigit(): mul = int(prefix)
            try: kb = float(setting) * mul; return kb / 1024.0
            except: return 0.0
        try: return float(setting)
        except: return 0.0

    def _record_trim(self, p: str, old_lo: float, old_hi: float, new_lo: float, new_hi: float, v: float, mode: str, eps: float):
        # (保持原樣 ...)
        if (old_lo, old_hi) != (new_lo, new_hi):
            self.last_trims.append(
                f"{p}:{mode} @{v:.3g} (−{eps:.3g}) {int(old_lo) if float(old_lo).is_integer() else old_lo:.3g}.."
                f"{int(old_hi) if float(old_hi).is_integer() else old_hi:.3g} → "
                f"{int(new_lo) if float(new_lo).is_integer() else new_lo:.3g}.."
                f"{int(new_hi) if float(new_hi).is_integer() else new_hi:.3g}"
            )

    def _fmt_num_for_log(self, p: str, x: float) -> str:
        # (保持原樣 ...)
        lo, hi = self.trust[p]["lo"], self.trust[p]["hi"]
        sample = self.param_specs[p]["cast"]((lo + hi) / 2.0)
        if isinstance(sample, (int, str)):
            try: return str(int(round(x)))
            except: return f"{x:.3g}"
        return f"{x:.3g}"

    def _schedule_probes_from_topk(self, topk: list[dict], bad_numeric: dict | None = None):
        # (保持原樣 ...)
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
        # (保持原樣 ...)
        p = pr["param"]
        good = float(pr["probe_numeric"])
        bad  = pr.get("bad_numeric", None)
        lo, hi = self.trust[p]["lo"], self.trust[p]["hi"]
        old_lo, old_hi = lo, hi
        
        spec = self.param_specs[p]
        sample = spec["cast"]((lo + hi) / 2.0)
        
        # 判斷是否為「類 Boolean」參數：
        # 1. 轉型後是字串 (舊邏輯，如 'on'/'off')
        # 2. 或者：轉型後是 int，且範圍剛好是 0 到 1 (P1 新邏輯)
        is_boolean_like = False
        if isinstance(sample, str):
            is_boolean_like = True
        elif isinstance(sample, int) and spec.get("min") == 0 and spec.get("max") == 1:
            is_boolean_like = True

        if is_boolean_like:
            # 執行鎖死 (Pin) 邏輯
            self.trust[p]["lo"] = self.trust[p]["hi"] = good
            self._fix_trust_bounds(p)
            self._record_trim(p, old_lo, old_hi, self.trust[p]["lo"], self.trust[p]["hi"], good, "probe→pin", 0.0)
            return
        # --- [修改結束] ---
        
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
        # (保持原樣 ...)
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
        
        # P2 關鍵修改: 在執行 SQL 前先套用 Session Parameters
        # 這樣就不需要重啟 DB
        
        for attempt in (1, 2):
            try:
                t0 = time.perf_counter()
                with self.conn.cursor() as cur:
                    # [NEW] Apply Session Params for P2
                    if self.tuning_mode == "session" and self.current_session_params:
                        for k, v in self.current_session_params.items():
                            # 使用 SET LOCAL 或 SET 都可以，這裡用 SET 確保在這個 transaction block 生效
                            # 注意: v 已經是 formatted string (e.g., 'on', '1024')
                            try:
                                cur.execute(f"SET {k} = {v}")
                            except Exception as set_e:
                                print(f"[Warning] Failed to SET {k}={v}: {set_e}")

                    # [Existing] Set Timeout
                    if timeout_ms is not None:
                        cur.execute("SET statement_timeout = %s", (int(timeout_ms),))
                    
                    try:
                        cur.execute(sql, params)
                        try: cur.fetchone()
                        except: pass
                        lat = (time.perf_counter() - t0) * 1000.0
                        if timeout_ms is not None: cur.execute("SET statement_timeout = 0")
                        return lat, False
                    except Exception as e:
                        # Timeout handling ...
                        msg = str(e)
                        if "statement timeout" in msg or "canceling statement" in msg or getattr(e, "pgcode", None) == "57014":
                            try: self.conn.rollback()
                            except: pass
                            try:
                                with self.conn.cursor() as c2: c2.execute("SET statement_timeout = 0")
                            except: pass
                            lat = float(timeout_ms if timeout_ms is not None else self.min_timeout_ms)
                            if lat <= 0: lat = self.min_timeout_ms
                            return lat, True
                        raise
            except psycopg2.OperationalError as e:
                # Reconnect logic ...
                msg = str(e)
                if attempt == 1 and ("SSL SYSCALL" in msg or "reset by peer" in msg or "connection not open" in msg or "closed" in msg):
                    print(f"[Warning] _execute_sql found closed connection (mode={self.tuning_mode}), reconnecting...")
                    self._reconnect_db()
                    continue
                raise

    def _total_cost(self):
        # 這裡也要確保 Explain 前有套用參數，所以我們可以簡單地呼叫 _execute_sql 的變體
        # 但為了簡單，我們在這裡重複 SET 的邏輯 (或者依賴 session 狀態)
        try:
            with self.conn.cursor() as cur: cur.execute("ROLLBACK")
        except:
            try: self.conn.rollback()
            except: pass
            
        with self.conn.cursor() as cur:
            # [NEW] P2 needs params for Explain too
            if self.tuning_mode == "session" and self.current_session_params:
                for k, v in self.current_session_params.items():
                    cur.execute(f"SET {k} = {v}")
                    
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
        
        # 這裡根據模式決定 reset 行為
        if self.start_from_default:
            self._apply_factory_defaults()
        
        # P2 session mode 也需要清空記憶
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
                    # (Assuming target parameters accept the same format, e.g., all integers)
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
        lat_ms, timed_out = self._execute_sql(timeout_ms=timeout_ms)
        
        # 4. Reward & Post-processing (Remain same ...)
        if timed_out:
            if self.probe_current is None:
                self.worst_numeric = dict(self.last_numeric_vals)
                ranked = self._rank_all_by_deviation(self.worst_numeric)
                
                # --- [修正開始] ---
                # 如果第一名是 Boolean 且偏差很大，就只處理它，不要把無辜的 Numeric 拖下水
                if ranked and ranked[0]["is_bool"] and abs(ranked[0]["delta"]) > 0.5:
                    # 只取所有 Boolean 類型的參數進入 Top-K
                    topk = [it for it in ranked if it["is_bool"]]
                    print(f"[TRIM Protection] Timeout caused by Flags. Scheduling only Boolean probes.")
                else:
                    # 否則才正常取前 K 個 (混合 Boolean 和 Numeric)
                    topk = ranked[: self.topk_k]
                # --- [修正結束] ---
                
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

        obs, cost = self._obs()

        if lat_ms / (self.latency_baseline_ms + 1e-6) > 1:
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
            # Session mode 下 pg_settings 也會反映 SET 的結果
            for p in self.tune_params:
                spec = self.param_specs[p]
                
                # [關鍵修正 1]: 處理虛擬參數
                if spec.get("is_virtual", False):
                    # 虛擬參數本身在 pg_settings 查不到，直接用我們剛剛算出的值
                    # 假設 self.last_numeric_vals 已經在前面被填入了 (這很重要!)
                    raw_val = self.last_numeric_vals.get(p, 0)
                    
                    # 2. [關鍵修正] 使用 spec['cast'] 強制轉回正確型別 (float -> int)
                    # 這樣 12.0 就會變回 12，才能被 {val:d} 格式化
                    cast_func = spec.get("cast", float)
                    val = cast_func(raw_val)
                    
                    # 3. 格式化
                    formatter = spec.get("fmt", "{val}")
                    if callable(formatter):
                        human_val = str(formatter(val))
                    else:
                        human_val = str(formatter.format(val=val))
                    
                    # 4. 填入 info
                    info[p] = human_val
                    cur_human[p] = human_val
                    cur_numeric[p] = float(val) # 這裡保持 float 給 RL 比較用沒關係
                    
                    # [關鍵修正 2]: 順便去查它底下的真實參數，並塞入 info
                    target_params = spec.get("map_to", [])
                    for tp in target_params:
                        cur.execute("SELECT setting, unit FROM pg_settings WHERE name = %s", (tp,))
                        res = cur.fetchone()
                        if res:
                            setting, unit = res
                            human = self._humanize_setting(setting, unit)
                            info[tp] = human # 讓真實參數出現在 Log 中
                            # 注意：這裡不需要加進 cur_human/numeric，因為 Agent 不需要知道它們

                else:
                    # [Original Logic]: 一般真實參數
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
        # (保持原樣 ...)
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
    
class ConvergenceStoppingCallback(BaseCallback):
    """
    When the environment's best latency does not improve by more than 'min_delta' for 'patience' steps, stop training.
    Includes a 'warmup_steps' period to ignore initial "fresh start" outliers.
    """
    def __init__(self, patience: int = 200, min_delta_ratio: float = 0.01, check_freq: int = 1, warmup_steps: int = 20, verbose: int = 1):
        super().__init__(verbose)
        self.patience = patience
        self.min_delta_ratio = min_delta_ratio
        self.check_freq = check_freq
        self.warmup_steps = warmup_steps
        
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
        current_step_latency = infos.get("latency", None)
        
        if current_step_latency is None:
             # Fallback to env best_latency_ms if not in infos
             current_step_latency = getattr(env, "best_latency_ms", None)

        if current_step_latency is None:
            return True

        # Initialize best_latency on first check after warmup
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