# param_specs_v9.py
# 3-phase hierarchical training parameter specifications for PgConfEnv
# Canonical parameter specifications and convenient presets for PgConfEnv.
# Each spec must define: min, max, fmt (a Python format string), and cast (a callable).
# fmt should render a literal suitable for `SET <guc> = <literal>`.
# Extended version with more parameters which may require a server restart.
# Added "virtual" parameter for parallel degree mapping.
# Added work_mem levels and shared_buffers levels for more realistic tuning.
# Removed parallel_degree and replaced with max_parallel_workers_per_gather.
# v9: Converted large continuous parameters (JIT, Cache, Mem, Scan Sizes) into discrete levels for faster RL convergence.

from typing import Dict, Tuple

# Helper for boolean on/off
_bool = lambda x: 'on' if int(round(x)) else 'off'

# Define discrete levels for certain parameters to speed up RL convergence and make it more interpretable.
PARALLEL_LEVELS = [0, 4, 8, 12, 16, 20]
WORK_MEM_LEVELS = [4, 8, 16, 32, 64, 128, 256]
SHARED_BUFFERS_LEVELS = [128, 256, 512, 1024, 2048, 4096, 8192]
EFFECTIVE_CACHE_LEVELS = [4096, 8192, 16384, 32768, 65536]
MAINTENANCE_WORK_MEM_LEVELS = [4, 8, 16, 32, 64]
AUTOVACUUM_WORK_MEM_LEVELS = [-1, 4, 8, 16, 32, 64]
DYNAMIC_SHARED_MEM_LEVELS = [0, 64, 128, 256, 512, 1024]
TABLE_SCAN_SIZE_LEVELS = [0, 8, 16, 32, 64, 128]
INDEX_SCAN_SIZE_LEVELS = [0, 4, 8, 16, 32, 64, 128, 256]
JIT_COST_LEVELS = [10000, 50000, 100000, 250000, 500000]

P1_PARAM_SPECS: Dict[str, dict] = {
    
    # --- RESOURCES_MEM (params count: 2) ---
    "work_mem": {
        "min": 0,
        "max": len(WORK_MEM_LEVELS) - 1,
        "fmt": lambda x, levels=WORK_MEM_LEVELS: f"{levels[int(round(x))]}MB",
        "cast": int,
        "levels": WORK_MEM_LEVELS,
        "levels_unit": "MB",
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
        "is_bool": True,
    },
    "max_parallel_workers_per_gather": {
        "min": 0,
        "max": len(PARALLEL_LEVELS) - 1,
        "fmt": lambda x, levels=PARALLEL_LEVELS: f"{levels[int(round(x))]}",
        "cast": int,
        "levels": PARALLEL_LEVELS,
        "levels_unit": "native",
    },

    # --- QUERY_TUNING_COST (params count: 10) ---
    "effective_cache_size": {
        "min": 0,
        "max": len(EFFECTIVE_CACHE_LEVELS) - 1,
        "fmt": lambda x, levels=EFFECTIVE_CACHE_LEVELS: f"{levels[int(round(x))]}MB",
        "cast": int,
        "levels": EFFECTIVE_CACHE_LEVELS,
        "levels_unit": "MB",
    },
    "cpu_tuple_cost": {
        "min": 0.01,
        "max": 0.1,            
        "fmt": "{val:.2f}",
        "cast": lambda v: round(float(v), 2),
    },
    "cpu_index_tuple_cost": {
        "min": 0.001,
        "max": 0.01,            
        "fmt": "{val:.3f}",
        "cast": lambda v: round(float(v), 3),
    },
    "cpu_operator_cost": {
        "min": 0.001,
        "max": 0.01,            
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
        "max": len(TABLE_SCAN_SIZE_LEVELS) - 1,
        "fmt": lambda x, levels=TABLE_SCAN_SIZE_LEVELS: f"{levels[int(round(x))]}MB",
        "cast": int,
        "levels": TABLE_SCAN_SIZE_LEVELS,
        "levels_unit": "MB",
    },
    "min_parallel_index_scan_size": {
        "min": 0,
        "max": len(INDEX_SCAN_SIZE_LEVELS) - 1,
        "fmt": lambda x, levels=INDEX_SCAN_SIZE_LEVELS: f"{levels[int(round(x))]}",
        "cast": int,
        "levels": INDEX_SCAN_SIZE_LEVELS,
        "levels_unit": "native",
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
        "is_bool": True,
    },
    "enable_indexscan": {
        "min": 0,
        "max": 1,
        "fmt": lambda x: _bool(x),
        "cast": int,
        "is_bool": True,
    },
    "enable_indexonlyscan": {
        "min": 0,
        "max": 1,
        "fmt": lambda x: _bool(x),
        "cast": int,
        "is_bool": True,
    },
    "enable_bitmapscan": {
        "min": 0,
        "max": 1,
        "fmt": lambda x: _bool(x),
        "cast": int,
        "is_bool": True,
    },
    "enable_sort": {
        "min": 0,
        "max": 1,
        "fmt": lambda x: _bool(x),
        "cast": int,
        "is_bool": True,
    },
    "enable_incremental_sort": {
        "min": 0,
        "max": 1,
        "fmt": lambda x: _bool(x),
        "cast": int,
        "is_bool": True,
    },
    "enable_hashagg": {
        "min": 0,
        "max": 1,
        "fmt": lambda x: _bool(x),
        "cast": int,
        "is_bool": True,
    },
    "enable_material": {
        "min": 0,
        "max": 1,
        "fmt": lambda x: _bool(x),
        "cast": int,
        "is_bool": True,
    },
    "enable_memoize": {
        "min": 0,
        "max": 1,
        "fmt": lambda x: _bool(x),
        "cast": int,
        "is_bool": True,
    },
    "enable_nestloop": {
        "min": 0,
        "max": 1,
        "fmt": lambda x: _bool(x),
        "cast": int,
        "is_bool": True,
    },
    "enable_mergejoin": {
        "min": 0,
        "max": 1,
        "fmt": lambda x: _bool(x),
        "cast": int,
        "is_bool": True,
    },
    "enable_hashjoin": {
        "min": 0,
        "max": 1,
        "fmt": lambda x: _bool(x),
        "cast": int,
        "is_bool": True,
    },
    "enable_gathermerge": {
        "min": 0,
        "max": 1,
        "fmt": lambda x: _bool(x),
        "cast": int,
        "is_bool": True,
    },
    "enable_parallel_hash": {
        "min": 0,
        "max": 1,
        "fmt": lambda x: _bool(x),
        "cast": int,
        "is_bool": True,
    },
    
}

P2_PARAM_SPECS: Dict[str, dict] = {
    
    # --- QUERY_TUNING_COST (params count: 3) ---
    "jit_above_cost": {
        "min": 0,
        "max": len(JIT_COST_LEVELS) - 1,
        "fmt": lambda x, levels=JIT_COST_LEVELS: f"{levels[int(round(x))]}",
        "cast": int,
        "levels": JIT_COST_LEVELS,
        "levels_unit": "native",
    },
    "jit_optimize_above_cost": {
        "min": 0,
        "max": len(JIT_COST_LEVELS) - 1,
        "fmt": lambda x, levels=JIT_COST_LEVELS: f"{levels[int(round(x))]}",
        "cast": int,
        "levels": JIT_COST_LEVELS,
        "levels_unit": "native",
    },
    "jit_inline_above_cost": {
        "min": 0,
        "max": len(JIT_COST_LEVELS) - 1,
        "fmt": lambda x, levels=JIT_COST_LEVELS: f"{levels[int(round(x))]}",
        "cast": int,
        "levels": JIT_COST_LEVELS,
        "levels_unit": "native",
    },
    
    # --- QUERY_TUNING_GEQO (params count: 6) ---
    "geqo": {
        "min": 0,
        "max": 1,
        "fmt": lambda x: _bool(x),
        "cast": int,
        "is_bool": True,
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
        "is_bool": True,
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
        "max": len(DYNAMIC_SHARED_MEM_LEVELS) - 1,
        "fmt": lambda x, levels=DYNAMIC_SHARED_MEM_LEVELS: f"{levels[int(round(x))]}MB",
        "cast": int,
        "levels": DYNAMIC_SHARED_MEM_LEVELS,
        "levels_unit": "MB",
    },
    "shared_buffers": {
        "min": 0,
        "max": len(SHARED_BUFFERS_LEVELS) - 1,
        "fmt": lambda x, levels=SHARED_BUFFERS_LEVELS: f"{levels[int(round(x))]}MB",
        "cast": int,
        "levels": SHARED_BUFFERS_LEVELS,
        "levels_unit": "MB",
    },
    "maintenance_work_mem": {
        "min": 0,
        "max": len(MAINTENANCE_WORK_MEM_LEVELS) - 1,
        "fmt": lambda x, levels=MAINTENANCE_WORK_MEM_LEVELS: f"{levels[int(round(x))]}MB",
        "cast": int,
        "levels": MAINTENANCE_WORK_MEM_LEVELS,
        "levels_unit": "MB",
    },
    "autovacuum_work_mem": {
        "min": 0,
        "max": len(AUTOVACUUM_WORK_MEM_LEVELS) - 1,
        "fmt": lambda x, levels=AUTOVACUUM_WORK_MEM_LEVELS: "-1" if levels[int(round(x))] == -1 else f"{levels[int(round(x))]}MB",
        "cast": int,
        "levels": AUTOVACUUM_WORK_MEM_LEVELS,
        "levels_unit": "MB",
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
PRESETS: Dict[str, Tuple[str, ...]] = {
    "basic3": ("work_mem", "random_page_cost", "max_parallel_workers_per_gather"),
    "mem_only": ("work_mem",),
    "planner_costs": ("random_page_cost","seq_page_cost","parallel_setup_cost","parallel_tuple_cost","effective_cache_size"),
    "parallel": ("max_parallel_workers_per_gather","min_parallel_table_scan_size","min_parallel_index_scan_size","parallel_leader_participation"),
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