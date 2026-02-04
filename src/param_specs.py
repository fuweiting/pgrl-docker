# param_specs_v5.py
# 3-phase hierarchical training parameter specifications for PgConfEnv
# Canonical parameter specifications and convenient presets for PgConfEnv.
# Each spec must define: min, max, fmt (a Python format string), and cast (a callable).
# fmt should render a literal suitable for `SET <guc> = <literal>`.
# Extended version with more parameters which may require a server restart.
# Added "virtual" parameter for parallel degree mapping.

from typing import Dict, Tuple

# Helper for boolean on/off
_bool = lambda x: 'on' if int(round(x)) else 'off'

PARALLEL_LEVELS = [0, 4, 8, 12, 16, 20]

P1_PARAM_SPECS: Dict[str, dict] = {
    
    # --- RESOURCES_MEM (params count: 2) ---
    "work_mem": {
        "min": 4,            # Min 4MB
        "max": 256,          # Max 256MB
        "fmt": "'{val:d}MB'",
        "cast": lambda x: int(round(x)),
    },
    "hash_mem_multiplier": {
        "min": 1.0,
        "max": 4.0,            
        "fmt": "{val:.1f}",
        "cast": lambda v: round(float(v), 1),
    },

    # --- RESOURCES_ASYNCHRONOUS (params count: 2) ---
    "parallel_leader_participation": {
        "min": 0,
        "max": 1,
        "fmt": lambda x: _bool(x),
        "cast": int,
    },
    # "max_parallel_workers_per_gather": {
    #     "min": 0,
    #     "max": 20,
    #     "fmt": "{val:d}",
    #     "cast": lambda x: int(round(x)),
    # },
    "parallel_degree": {
        "is_virtual": True,
        "map_to": ["max_parallel_workers_per_gather"], # 實際控制的參數
        "min": 0,
        "max": 5, # 對應 PARALLEL_LEVELS 的 index 0~5
        "fmt": lambda x: PARALLEL_LEVELS[int(round(x))], # 將 0~5 轉為 0, 4...20
        "cast": int,
    },

    # --- QUERY_TUNING_COST (params count: 10) ---
    "effective_cache_size": {
        "min": 4096,      # Min 4GB
        "max": 65536,     # Max 64GB         
        "fmt": "'{val:d}MB'",
        "cast": lambda x: int(round(x)),
    },
    "cpu_tuple_cost": {
        "min": 0.01,
        "max": 0.5,            
        "fmt": "{val:.2f}",
        "cast": lambda v: round(float(v), 2),
    },
    "cpu_index_tuple_cost": {
        "min": 0.001,
        "max": 0.1,            
        "fmt": "{val:.3f}",
        "cast": lambda v: round(float(v), 3),
    },
    "cpu_operator_cost": {
        "min": 0.001,
        "max": 0.1,            
        "fmt": "{val:.3f}",
        "cast": lambda v: round(float(v), 3),
    },
    "seq_page_cost": {
        "min": 1,
        "max": 4,            
        "fmt": "{val:.2f}",
        "cast": lambda v: round(float(v), 2),
    },
    "random_page_cost": {
        "min": 1,
        "max": 4,            
        "fmt": "{val:.2f}",
        "cast": lambda v: round(float(v), 2),
    },
    "min_parallel_table_scan_size": {
        "min": 0,
        "max": 128,
        "fmt": "'{val:d}MB'",
        "cast": lambda x: int(round(x)),
    },
    "min_parallel_index_scan_size": {
        "min": 0,
        "max": 256,
        "fmt": "{val:d}",
        "cast": lambda x: int(round(x)),
    },
    "parallel_tuple_cost": {
        "min": 0,
        "max": 1,
        "fmt": "{val:.2f}",
        "cast": lambda v: round(float(v), 2),
    },
    "parallel_setup_cost": {
        "min": 0,
        "max": 1000,
        "fmt": "{val:d}",
        "cast": lambda x: int(round(x)),
    },
    
    # --- QUERY_TUNING_METHOD (params count: 14) ---
    "enable_seqscan": {
        "min": 0,
        "max": 1,
        "fmt": lambda x: _bool(x),
        "cast": int,
    },
    "enable_indexscan": {
        "min": 0,
        "max": 1,
        "fmt": lambda x: _bool(x),
        "cast": int,
    },
    "enable_indexonlyscan": {
        "min": 0,
        "max": 1,
        "fmt": lambda x: _bool(x),
        "cast": int,
    },
    "enable_bitmapscan": {
        "min": 0,
        "max": 1,
        "fmt": lambda x: _bool(x),
        "cast": int,
    },
    "enable_sort": {
        "min": 0,
        "max": 1,
        "fmt": lambda x: _bool(x),
        "cast": int,
    },
    "enable_incremental_sort": {
        "min": 0,
        "max": 1,
        "fmt": lambda x: _bool(x),
        "cast": int,
    },
    "enable_hashagg": {
        "min": 0,
        "max": 1,
        "fmt": lambda x: _bool(x),
        "cast": int,
    },
    "enable_material": {
        "min": 0,
        "max": 1,
        "fmt": lambda x: _bool(x),
        "cast": int,
    },
    "enable_memoize": {
        "min": 0,
        "max": 1,
        "fmt": lambda x: _bool(x),
        "cast": int,
    },
    "enable_nestloop": {
        "min": 0,
        "max": 1,
        "fmt": lambda x: _bool(x),
        "cast": int,
    },
    "enable_mergejoin": {
        "min": 0,
        "max": 1,
        "fmt": lambda x: _bool(x),
        "cast": int,
    },
    "enable_hashjoin": {
        "min": 0,
        "max": 1,
        "fmt": lambda x: _bool(x),
        "cast": int,
    },
    "enable_gathermerge": {
        "min": 0,
        "max": 1,
        "fmt": lambda x: _bool(x),
        "cast": int,
    },
    "enable_parallel_hash": {
        "min": 0,
        "max": 1,
        "fmt": lambda x: _bool(x),
        "cast": int,
    },
    
}

P2_PARAM_SPECS: Dict[str, dict] = {
    
    # --- QUERY_TUNING_COST (params count: 3) ---
    "jit_above_cost": {
        "min": 10000,
        "max": 200000,
        "fmt": "{val:d}",
        "cast": lambda x: int(round(x)),
    },
    "jit_optimize_above_cost": {
        "min": 10000,
        "max": 500000,
        "fmt": "{val:d}",
        "cast": lambda x: int(round(x)),
    },
    "jit_inline_above_cost": {
        "min": 10000,
        "max": 500000,
        "fmt": "{val:d}",
        "cast": lambda x: int(round(x)),
    },
    
    # --- QUERY_TUNING_GEQO (params count: 6) ---
    "geqo": {
        "min": 0,
        "max": 1,
        "fmt": lambda x: _bool(x),
        "cast": int,
    },
    "geqo_threshold": {
        "min": 2,
        "max": 14,
        "fmt": "{val:d}",
        "cast": lambda x: int(round(x)),
    },
    "geqo_effort": {
        "min": 1,
        "max": 10,
        "fmt": "{val:d}",
        "cast": lambda x: int(round(x)),
    },
    "geqo_pool_size": {
        "min": 0,
        "max": 1000,
        "fmt": "{val:d}",
        "cast": lambda x: int(round(x)),
    },
    "geqo_generations": {
        "min": 0,
        "max": 100,
        "fmt": "{val:d}",
        "cast": lambda x: int(round(x)),
    },
    "geqo_selection_bias": {
        "min": 1.5,
        "max": 2.0,
        "fmt": "{val:.2f}",
        "cast": lambda v: round(float(v), 2),
    },
    
    # --- QUERY_TUNING_OTHER (params count: 3) ---
    "jit": {
        "min": 0,
        "max": 1,
        "fmt": lambda x: _bool(x),
        "cast": int,
    },
    "from_collapse_limit": {
        "min": 1,
        "max": 18,
        "fmt": "{val:d}",
        "cast": lambda x: int(round(x)),
    },
    "join_collapse_limit": {
        "min": 1,
        "max": 18,
        "fmt": "{val:d}",
        "cast": lambda x: int(round(x)),
    },

}

P3_PARAM_SPECS: Dict[str, dict] = {
    
    # --- RESOURCES_MEM (params count: 4) ---
    "min_dynamic_shared_memory": {
        "min": 0,              
        "max": 1024,            # 0 MB ~ 1 GB
        "fmt": "{val:d}MB",   # 加上單引號與 MB
        "cast": lambda x: int(round(x)),
    },
    "shared_buffers": {
        # 原本 8192 blocks (x8kB) = 64MB
        # 原本 1048576 blocks (x8kB) = 8192MB (8GB)
        "min": 64,              # 64 MB
        "max": 8192,            # 8 GB
        "fmt": "{val:d}MB",
        "cast": lambda x: int(round(x)),
    },
    "maintenance_work_mem": {
        "min": 4,               # 4 MB
        "max": 64,              # 64 MB
        "fmt": "{val:d}MB",
        "cast": lambda x: int(round(x)),
    },
    "autovacuum_work_mem": {
        "min": -1,
        "max": 64,
        # [特殊處理]: 如果值 < 1 (即 -1 或 0)，輸出 '-1' (無單位)
        # 如果值 >= 1，輸出 'xMB' (有單位)
        "fmt": lambda x: "-1" if x < 1 else f"{int(x)}MB",
        "cast": lambda x: int(round(x)),
    },
    
    # --- RESOURCES_ASYNCHRONOUS (params count: 2) ---
    "effective_io_concurrency": {
        "min": 1,
        "max": 200,
        "fmt": "{val:d}",
        "cast": lambda x: int(round(x)),
    },
    "maintenance_io_concurrency": {
        "min": 1,
        "max": 200,
        "fmt": "{val:d}",
        "cast": lambda x: int(round(x)),
    },

}

PARAM_SPECS = {**P1_PARAM_SPECS, **P2_PARAM_SPECS, **P3_PARAM_SPECS}

# Common selections you can reuse by name in scripts
# parallel_degree is a "virtual" param mapped to max_parallel_workers_per_gather
PRESETS: Dict[str, Tuple[str, ...]] = {
    "basic3": ("work_mem", "random_page_cost", "parallel_degree"),
    "mem_only": ("work_mem",),
    "planner_costs": ("random_page_cost","seq_page_cost","parallel_setup_cost","parallel_tuple_cost","effective_cache_size"),
    "parallel": ("parallel_degree","min_parallel_table_scan_size","min_parallel_index_scan_size","parallel_leader_participation"),
    "enablers": ("enable_hashjoin","enable_mergejoin","enable_nestloop","enable_indexscan","enable_indexonlyscan","enable_bitmapscan","enable_gathermerge","enable_parallel_hash","enable_memoize","enable_hashagg","enable_incremental_sort","enable_seqscan"),
    "jit": ("jit","jit_above_cost","jit_inline_above_cost","jit_optimize_above_cost"),
    "join_search": ("join_collapse_limit","from_collapse_limit","geqo","geqo_threshold"),
    "memory": ("work_mem","effective_cache_size","maintenance_work_mem","hash_mem_multiplier"),
}

def select_params(*, params_csv: str = None, preset: str = None, all_params: bool = False, available: Dict[str, dict] = None):
    """Return a tuple of parameter names to tune.
    
    Priority: `all_params` > `preset` > `params_csv` > default (all).
    - params_csv: comma-separated names (e.g., "work_mem,random_page_cost")
    - preset: a key in PRESETS (e.g., "basic3")
    - all_params: when True, return all available parameters.
    """
    specs = available or PARAM_SPECS
    if all_params:
        return tuple(specs.keys())
    if preset:
        if preset not in PRESETS:
            raise KeyError(f"Unknown preset '{preset}'. Available: {list(PRESETS)}")
        return PRESETS[preset]
    if params_csv:
        names = tuple(p.strip() for p in params_csv.split(',') if p.strip())
        missing = [n for n in names if n not in specs]
        if missing:
            raise KeyError(f"Unknown params {missing}. Available: {list(PARAM_SPECS)}")
        return names
    # default: all known params
    return tuple(specs.keys())
