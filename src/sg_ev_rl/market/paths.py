"""Locations of the research data (read-only) and of large run outputs."""
import os
from pathlib import Path

DEFAULT_RESEARCH_ROOT = "/Users/yyx/Library/CloudStorage/OneDrive-NanyangTechnologicalUniversity/Research"
DEFAULT_RUNS_DIR = "/Users/yyx/Documents/GitHub/sg_ev_aggregator_rl_runs/c"


def research_root() -> Path:
    return Path(os.environ.get("SG_RESEARCH_ROOT", DEFAULT_RESEARCH_ROOT))


def datasets_dir() -> Path:
    return research_root() / "datasets"


def market_table_path() -> Path:
    return datasets_dir() / "interim" / "market_halfhourly.parquet"


def raw_market_dir() -> Path:
    return datasets_dir() / "raw" / "sg_grid_market"


def runs_dir() -> Path:
    return Path(os.environ.get("SG_RUNS_DIR_C", DEFAULT_RUNS_DIR))
